"""
Triton compilation strategy implementation
"""

import ast
from dataclasses import dataclass
import logging
import time
import torch
import triton
import triton.language as tl
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import importlib.util, tempfile, pathlib, sys

from compilation.base_compiler import BaseCompilationBackend
from shared.models import ArgSpec, BaseExecutableKernel, KernelCode, IOContract, KernelType, TensorSpec
from .triton_extractor import TritonKernelExtractor, TritonKernelInfo
from shared.utils import materialize_tensor

logger = logging.getLogger(__name__)


@dataclass
class _CapturedArg:
    """Internal runtime snapshot of an argument. Convert to ArgSpec later."""
    name: str
    kind: str  # "tensor" | "int" | "float" | "bool" | "str" | "other"
    value: Optional[Union[int, float, str, bool]] = None
    tensor_spec: Optional[TensorSpec] = None


@dataclass
class _KernelLaunchCapture:
    """Internal snapshot of a single Triton kernel launch"""
    kernel_name: str
    grid_obj: Any
    resolved_grid: Optional[Union[int, Tuple[int, ...]]]
    meta: Dict[str, Any]
    pos_args: List[_CapturedArg]
    kw_args: Dict[str, _CapturedArg]


# --- Utilities ---

def _sync_all():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _torch_dtype_to_str(dt: torch.dtype) -> str:
    # Map torch dtype to strings used in TensorSpec
    mapping = {
        torch.float64: "float64",
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.int64: "int64",
        torch.int32: "int32",
        torch.int16: "int16",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    return mapping.get(dt, str(dt).split(".")[-1])  # fallback

def _str_to_torch_dtype(s: str) -> torch.dtype:
    rev = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if s not in rev:
        raise ValueError(f"Unsupported dtype: {s}")
    return rev[s]


def _find_jitfunction_class():
    if triton is None:
        raise RuntimeError("Triton is not installed.")
    try:
        import triton.runtime.jit as trjit
        if hasattr(trjit, "JITFunction"):
            return trjit.JITFunction
    except Exception:
        pass
    try:
        if hasattr(triton, "runtime") and hasattr(triton.runtime, "JITFunction"):
            return triton.runtime.JITFunction
    except Exception:
        pass
    for name in dir(triton):
        obj = getattr(triton, name, None)
        if isinstance(obj, type) and hasattr(obj, "__getitem__") and hasattr(obj, "__call__"):
            return obj
    raise RuntimeError("Could not locate Triton's JITFunction class.")


def _describe_runtime_arg(name: str, x: Any) -> _CapturedArg:
    if torch.is_tensor(x):
        return _CapturedArg(
            name=name,
            kind="tensor",
            tensor_spec=TensorSpec(
                shape=[int(s) for s in x.shape],
                dtype=_torch_dtype_to_str(x.dtype)
            )
        )
    if isinstance(x, bool):
        return _CapturedArg(name=name, kind="bool", value=bool(x))
    if isinstance(x, int):
        return _CapturedArg(name=name, kind="int", value=int(x))
    if isinstance(x, float):
        return _CapturedArg(name=name, kind="float", value=float(x))
    if isinstance(x, str):
        return _CapturedArg(name=name, kind="str", value=x)
    return _CapturedArg(name=name, kind="other")


def _eval_grid_if_callable(grid_obj: Any, meta: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    """
    Resolve a grid object (int | tuple/list[int] | callable(META)->int|tuple[int,...])
    into a tuple[int,...] for Triton 3.x. Returns None if it cannot be resolved.
    """
    def _to_tuple(g):
        if g is None:
            return None
        if isinstance(g, int):
            return (int(g),)
        if isinstance(g, (tuple, list)):
            return tuple(int(v) for v in g)
        return None

    if callable(grid_obj):
        class _MetaProxy(dict):
            def __getattr__(self, k): return self[k]
        META = _MetaProxy(meta or {})
        try:
            resolved = grid_obj(META)
            return _to_tuple(resolved)
        except Exception:
            return None

    return _to_tuple(grid_obj)


def _as_triton_grid_tuple(grid_like: Any, meta: Optional[Dict[str, Any]] = None) -> Tuple[int, ...]:
    """
    Best-effort conversion of grid_like (int | seq | callable) to a tuple[int,...].
    Falls back to (1,) if not resolvable.
    """
    if callable(grid_like):
        resolved = _eval_grid_if_callable(grid_like, meta or {})
        return resolved if resolved is not None else (1,)
    if isinstance(grid_like, int):
        return (int(grid_like),)
    if isinstance(grid_like, (tuple, list)):
        return tuple(int(x) for x in grid_like)
    if grid_like is None:
        return (1,)
    # last resort
    try:
        return tuple(int(x) for x in grid_like)  # in case it's some sequence-like
    except Exception:
        return (1,)


class _TritonRuntimeInterceptor:
    """
    Context manager that patches Triton JITFunction __getitem__ and __call__
    to capture real launches (grid/meta/args).

    IMPORTANT: We return a wrapper callable from __getitem__ instead of mutating
    launcher.__call__, because __call__ special method lookup is on the class.
    """
    def __init__(self, jit_class=None, target_kernel_name: Optional[str] = None):
        self._JIT = jit_class or _find_jitfunction_class()
        self._orig_getitem = None
        self._orig_call = None
        self._captures: List[_KernelLaunchCapture] = []
        self._target = target_kernel_name

    def __enter__(self):
        J = self._JIT
        self._orig_getitem = J.__getitem__
        self._orig_call = J.__call__

        def _kernel_name(jitfunc):
            try:
                return getattr(jitfunc, "fn").__name__
            except Exception:
                return getattr(jitfunc, "__name__", "unknown_kernel")

        def patched_getitem(jitfunc, grid_obj):
            # Get the original launcher (callable object)
            launcher = self._orig_getitem(jitfunc, grid_obj)

            # Return a plain Python function that wraps the call
            def wrapped_launcher(*args, **kwargs):
                kname = _kernel_name(jitfunc)
                if (self._target is None) or (kname == self._target):
                    # Split kwargs into META (constexpr + num_warps/stages) vs data kwargs
                    meta_keys = {k for k in kwargs if (k.isupper() or k in ("num_warps", "num_stages", "num_ctas"))}
                    meta = {k: kwargs[k] for k in meta_keys}
                    data_kwargs = {k: kwargs[k] for k in kwargs if k not in meta_keys}

                    pos = [_describe_runtime_arg(f"arg{i}", a) for i, a in enumerate(args)]
                    kw = {k: _describe_runtime_arg(k, v) for k, v in data_kwargs.items()}
                    resolved = _eval_grid_if_callable(grid_obj, meta)

                    self._captures.append(_KernelLaunchCapture(
                        kernel_name=kname,
                        grid_obj=grid_obj,
                        resolved_grid=resolved,
                        meta=meta,
                        pos_args=pos,
                        kw_args=kw
                    ))
                return launcher(*args, **kwargs)

            return wrapped_launcher

        def patched_call(jitfunc, *args, **kwargs):
            # Rare direct-call path: kernel(*args, **kwargs)
            kname = _kernel_name(jitfunc)
            if (self._target is None) or (kname == self._target):
                meta_keys = {k for k in kwargs if (k.isupper() or k in ("num_warps", "num_stages", "num_ctas"))}
                meta = {k: kwargs[k] for k in meta_keys}
                data_kwargs = {k: kwargs[k] for k in kwargs if k not in meta_keys}
                pos = [_describe_runtime_arg(f"arg{i}", a) for i, a in enumerate(args)]
                kw = {k: _describe_runtime_arg(k, v) for k, v in data_kwargs.items()}

                self._captures.append(_KernelLaunchCapture(
                    kernel_name=kname,
                    grid_obj=1,
                    resolved_grid=1,
                    meta=meta,
                    pos_args=pos,
                    kw_args=kw
                ))
            return self._orig_call(jitfunc, *args, **kwargs)

        J.__getitem__ = patched_getitem
        J.__call__ = patched_call
        return self

    def __exit__(self, exc_type, exc, tb):
        J = self._JIT
        if self._orig_getitem:
            J.__getitem__ = self._orig_getitem
        if self._orig_call:
            J.__call__ = self._orig_call
        return False

    @property
    def captures(self) -> List[_KernelLaunchCapture]:
        return list(self._captures)

# --- Module loader for Triton source ---

class _TritonModuleLoader:
    def __init__(self, source: str):
        self.source = source
        self.ns: Dict[str, Any] = {}
        self._temp_dir = None
        self._temp_path = None
        
    def exec(self):
        self._temp_dir = tempfile.mkdtemp(prefix="triton_kernel_")
        self._temp_path = pathlib.Path(self._temp_dir) / "user_triton_module.py"
        self._temp_path.write_text(self.source, encoding="utf-8")

        import hashlib, importlib.util
        source_hash = hashlib.md5(self.source.encode()).hexdigest()[:8]
        module_name = f"user_triton_module_{source_hash}"
        
        spec = importlib.util.spec_from_file_location(module_name, self._temp_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        
        self.ns = mod.__dict__
        return self
    
    def cleanup(self):
        if self._temp_dir and pathlib.Path(self._temp_dir).exists():
            import shutil
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
            self._temp_path = None
    
    def __del__(self):
        try:
            self.cleanup()
        except:
            pass

    def list_kernels(self) -> List[Tuple[str, Any]]:
        out: List[Tuple[str, Any]] = []
        try:
            import triton.runtime.jit as trjit
            JIT = trjit.JITFunction
        except Exception:
            JIT = None

        for k, v in self.ns.items():
            if JIT is not None and isinstance(v, JIT):
                out.append((k, v))
                continue
            # very conservative fallback (avoid regular functions)
            if hasattr(v, "__getitem__") and hasattr(v, "__call__") and hasattr(v, "fn"):
                out.append((k, v))
        return out


class TritonExecutableKernel(BaseExecutableKernel):
    """
    Executable Triton kernel that:
      - loads user source code,
      - captures a real kernel launch (grid/meta/args),
      - can compile/run with synthetic inputs,
      - can do correctness checks and profiling (CUDA Events & NCU).
    """
    def __init__(
        self,
        kernel_code: KernelCode,
        device: torch.device,
        kernel_name: Optional[str] = None,
    ):
        assert kernel_code.kernel_type == KernelType.TRITON, "TritonExecutableKernel requires KernelType.TRITON"
        self.kernel_code = kernel_code
        self.kernel_name = kernel_name  # if None, we’ll pick the first discovered
        self._loader: Optional[_TritonModuleLoader] = None
        self._jit_func = None
        self._captures: List[_KernelLaunchCapture] = []
        super().__init__(kernel_type=kernel_code.kernel_type, device=device, io_contract=kernel_code.io)

    def _initialize_kernel(self):
        self._loader = _TritonModuleLoader(self.kernel_code.source_code).exec()
        kernels = self._loader.list_kernels()

        if not kernels:
            raise RuntimeError("No @triton.jit kernels found in source.")
        if self.kernel_name:
            for k, v in kernels:
                if k == self.kernel_name:
                    self._jit_func = v
                    break
            if self._jit_func is None:
                raise RuntimeError(f"Kernel '{self.kernel_name}' not found. Available: {[k for k,_ in kernels]}")
        else:
            self.kernel_name, self._jit_func = kernels[0]

    def _build_args_from_io(
        self,
        io: IOContract,
        provided_inputs: Tuple[Any, ...]
    ) -> Tuple[List[Any], Dict[str, Any], Dict[str, Any], List[int]]:
        """
        Build positional args in IOContract.args order (excluding is_meta=True),
        and gather meta kwargs from ArgSpec(is_meta=True).
        Returns: (positional_args, ctx, meta_kwargs, output_indices)
        """
        args: List[Any] = []
        ctx: Dict[str, Any] = {}
        out_idx: List[int] = []

        # Queue user-provided tensor inputs in order for tensor ArgSpecs with role in {"input","inout"}
        provided_tensors = [x for x in provided_inputs if torch.is_tensor(x)]
        pt_i = 0

        for i, a in enumerate(io.args):
            # ---- constexpr/meta MUST NOT be passed positionally ----
            if getattr(a, "is_meta", False):
                # Still expose meta to ctx for grid heuristics
                if a.value is not None:
                    ctx[a.name] = a.value
                continue

            if a.type == "tensor":
                if a.role in ("input", "inout"):
                    if pt_i < len(provided_tensors):
                        t = provided_tensors[pt_i]; pt_i += 1
                    else:
                        if not a.tensor_spec:
                            raise RuntimeError(f"Tensor arg '{a.name}' requires tensor_spec to allocate")
                        t = materialize_tensor(a.tensor_spec, default_device=self.device)
                        if t.device != self.device:
                            t = t.to(self.device)
                    args.append(t)
                    ctx[a.name] = t
                elif a.role == "output":
                    if not a.tensor_spec:
                        raise RuntimeError(f"Output tensor '{a.name}' requires tensor_spec (shape/dtype/device)")
                    t = materialize_tensor(a.tensor_spec, default_device=self.device)
                    if t.device != self.device:
                        t = t.to(self.device)
                    args.append(t)
                    out_idx.append(i)
                    ctx[a.name] = t
                else:
                    raise ValueError(f"Unknown tensor role for arg '{a.name}': {a.role}")
            else:
                # scalar (non-meta)
                v = int(a.value) if isinstance(a.value, bool) else a.value
                args.append(v)
                ctx[a.name] = v

        # Triton meta kwargs from ArgSpec(is_meta=True)
        meta: Dict[str, Any] = {a.name: a.value for a in io.args if getattr(a, "is_meta", False)}

        # Include num_warps/num_stages if set
        if io.launch and io.launch.num_warps is not None:
            meta["num_warps"] = int(io.launch.num_warps)
        if io.launch and io.launch.num_stages is not None:
            meta["num_stages"] = int(io.launch.num_stages)

        return args, ctx, meta, out_idx


    def _resolve_triton_grid(self, io: Optional[IOContract]) -> Union[Tuple[int], Tuple[int,int], Tuple[int,int,int]]:
        # explicit grid provided
        if io and io.launch and io.launch.grid:
            g = io.launch.grid
            # Keep as 3D tuple; Triton is fine with (gx, gy, gz) even if some are 1
            return (int(g.x), int(g.y), int(g.z))
        # fallback
        return (1,)
    
    # --- Capture ---

    def capture_once(self, invoke: Callable[[], Any]) -> IOContract:
        """
        Run user-supplied invoke() that launches the kernel once.
        Captures grid/meta/args and returns an IOContract derived from the call.
        """
        # Ensure loader / jit func are set up
        if self._loader is None or self._jit_func is None:
            self._loader = _TritonModuleLoader(self.kernel_code.source_code).exec()
            kernels = self._loader.list_kernels()
            if not kernels:
                raise RuntimeError("No @triton.jit kernels found in source.")
            if self.kernel_name:
                for k, v in kernels:
                    if k == self.kernel_name:
                        self._jit_func = v
                        break
                if self._jit_func is None:
                    raise RuntimeError(f"Kernel '{self.kernel_name}' not found. Available: {[k for k,_ in kernels]}")
            else:
                self.kernel_name, self._jit_func = kernels[0]

        # Patch the *class* of this jit function so our instance is affected
        JITClass = type(self._jit_func)

        with _TritonRuntimeInterceptor(jit_class=JITClass, target_kernel_name=self.kernel_name) as inter:
            invoke()

        # Filter to our kernel and take the last capture
        filtered = [c for c in inter.captures if c.kernel_name == self.kernel_name]
        if not filtered:
            raise RuntimeError(f"No launches captured for kernel '{self.kernel_name}'.")
        cap = filtered[-1]
        self._captures.append(cap)

        # Build ArgSpecs from captured positional & keyword data-args
        args_specs: List[ArgSpec] = []
        for carg in cap.pos_args:
            if carg.kind == "tensor" and carg.tensor_spec:
                args_specs.append(ArgSpec(name=carg.name, type="tensor", tensor_spec=carg.tensor_spec))
            elif carg.kind in {"int","float","bool","str"}:
                args_specs.append(ArgSpec(name=carg.name, type=carg.kind, value=carg.value))
            else:
                args_specs.append(ArgSpec(name=carg.name, type="str", value=str(carg.value)))

        for k, carg in cap.kw_args.items():
            if carg.kind == "tensor" and carg.tensor_spec:
                args_specs.append(ArgSpec(name=k, type="tensor", tensor_spec=carg.tensor_spec))
            elif carg.kind in {"int","float","bool","str"}:
                args_specs.append(ArgSpec(name=k, type=carg.kind, value=carg.value))
            else:
                args_specs.append(ArgSpec(name=k, type="str", value=str(carg.value)))

        # Meta (constexpr) -> ArgSpec with is_meta=True
        for k, v in cap.meta.items():
            t = "int" if isinstance(v, int) else "float" if isinstance(v, float) else "str"
            args_specs.append(ArgSpec(name=k, type=t, value=v, is_meta=True))

        # Update this instance’s specs and return the IOContract
        self.input_specs = args_specs
        if self.output_specs is None:
            self.output_specs = []
        return IOContract(args=args_specs, outputs=self.output_specs)
    # --- Execution ---

    def _select_output_indices(self) -> List[int]:
        if getattr(self, "output_specs", None) is None:
            self.output_specs = []
        if hasattr(self, "input_specs") and self.input_specs:
            if hasattr(self, "output_arg_indices") and self.output_arg_indices:
                return list(self.output_arg_indices)

            # Try ArgSpec.is_output
            idxs = [i for i, a in enumerate(self.input_specs) if getattr(a, "role") in ["output", "inout"]]
            if idxs:
                return idxs

        return []  # nothing marked; return nothing by default

    def _build_concrete_args_from_specs(
        self, provided_inputs: Tuple[Any, ...]
    ) -> Tuple[List[Any], List[int]]:
        """
        Build the full arg list to call the kernel:
        - Use user-provided inputs when given.
        - Allocate any missing tensor args from ArgSpec.tensor_spec.
        - Return the indices that should be considered outputs.
        """
        if not self.input_specs:
            raise RuntimeError("No input_specs available; capture_once() or provide IOContract first.")

        out_indices = self._select_output_indices()
        args: List[Any] = []
        p = list(provided_inputs)
        p_i = 0

        for i, spec in enumerate(self.input_specs):
            # NEVER put constexpr/meta into positional args
            if getattr(spec, "is_meta", False):
                continue

            if p_i < len(p):
                args.append(p[p_i]); p_i += 1
                continue

            if spec.type == "tensor" and spec.tensor_spec:
                ts = spec.tensor_spec
                dtype = _str_to_torch_dtype(ts.dtype)
                dev = torch.device(ts.device) if ts.device else self.device
                buf = torch.empty(ts.shape, dtype=dtype, device=dev)
                args.append(buf)
            elif spec.type in {"int","float","bool","str"}:
                args.append(spec.value)
            else:
                args.append(None)

        return args, out_indices

    def _return_outputs(self, args: List[Any], out_indices: List[int]):
        if not out_indices:
            # Nothing declared as outputs; return all tensors (safe fallback) OR return tuple(args)
            return [x for x in args if torch.is_tensor(x)]
        outputs = [args[i] for i in out_indices]
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
        
    def _spec_by_name(self, name: str):
        if not getattr(self, "input_specs", None):
            return None
        for s in self.input_specs:
            if s.name == name:
                return s
        return None

    def _materialize_from_captured(self, carg: _CapturedArg):
        # Prefer a matching ArgSpec (client-provided) to decide dtype/shape/device
        spec = self._spec_by_name(carg.name)
        if carg.kind == "tensor":
            # Choose spec if present; else use captured tensor_spec
            ts = spec.tensor_spec if (spec and spec.tensor_spec) else carg.tensor_spec
            assert ts is not None, f"No tensor_spec available to materialize '{carg.name}'"
            dt = _str_to_torch_dtype(ts.dtype)
            dev = self.device
            return torch.empty(ts.shape, dtype=dt, device=dev)
        if carg.kind in {"int", "float", "bool", "str"}:
            # Use explicit ArgSpec.value if provided; else captured literal
            if spec and spec.value is not None:
                return int(spec.value) if carg.kind in {"int", "bool"} else spec.value
            return int(carg.value) if carg.kind in {"int", "bool"} else carg.value
        # fallback
        spec = self._spec_by_name(carg.name)
        return None if spec is None else spec.value    

    def _execute_impl(self, *inputs):
        """
        If we have a capture, replay its calling convention.
        If not, fall back to IOContract-based path.
        """
        if self._captures:
            cap = self._captures[-1]
            raw_grid = cap.resolved_grid if (cap and cap.resolved_grid is not None) else cap.grid_obj
            grid = _as_triton_grid_tuple(raw_grid, cap.meta)
            
            if inputs:
                # call with provided inputs but preserve kwargs/meta separation
                # build data_kwargs from captured kw-args; meta from cap.meta
                _, data_kwargs, meta_kwargs, out_indices = self._reconstruct_from_capture(cap)
                # Replace positional tensors with user inputs (assumes same count/order as captured)
                pos = list(inputs)
                self._jit_func[grid](*pos, **data_kwargs, **meta_kwargs)
                return self._return_outputs(pos, out_indices)
            else:
                pos, data_kwargs, meta_kwargs, out_indices = self._reconstruct_from_capture(cap)
                self._jit_func[grid](*pos, **data_kwargs, **meta_kwargs)
                return self._return_outputs(pos, out_indices)

        # No capture: use IOContract path
        io = self.kernel_code.io
        pos_args, ctx, meta, out_idx = self._build_args_from_io(io, inputs)
        grid = self._resolve_triton_grid(io)
        self._jit_func[grid](*pos_args, **meta)
        return self._return_outputs(pos_args, out_idx)

    def _reconstruct_from_capture(self, cap: _KernelLaunchCapture):
        """
        Rebuild the exact calling pattern of the captured launch:
        - positional args in cap.pos_args order (excluding meta)
        - data kwargs from cap.kw_args (non-meta)
        - meta kwargs from cap.meta (constexpr/num_warps/stages)
        - output indices among *positional* args (by ArgSpec.role if available)
        """
        # positional args
        pos_args = []
        out_indices = []
        for i, carg in enumerate(cap.pos_args):
            # if the *spec* for this name is meta, skip from pos_args (constexprs must be kwargs)
            spec = self._spec_by_name(carg.name)
            if spec is not None and getattr(spec, "is_meta", False):
                continue
            val = self._materialize_from_captured(carg)
            pos_args.append(val)
            # mark outputs by ArgSpec.role if provided
            if spec is not None and getattr(spec, "role", None) in ("output", "inout"):
                out_indices.append(i if len(out_indices) == 0 else len(pos_args) - 1)

        # data kwargs (non-meta keyword args at call site)
        data_kwargs = {}
        for k, carg in cap.kw_args.items():
            spec = self._spec_by_name(k)
            if spec is not None and getattr(spec, "is_meta", False):
                # meta should not be in data kwargs
                continue
            data_kwargs[k] = self._materialize_from_captured(carg)

        # meta kwargs exactly as captured (constexpr + num_warps/stages)
        meta_kwargs = dict(cap.meta)

        return pos_args, data_kwargs, meta_kwargs, out_indices
    

    # --- High-level tasks required ---

    # (2) Compilation check: compile & run with synthetic inputs
    def compile_and_run(self, fill: str = "randn"):
        """
        Smoke compile + run with synthesized inputs. If we have a capture, we
        reconstruct the exact calling convention so constexpr/meta don't end up as positional.
        """
        cap = self._captures[-1] if self._captures else None

        if cap:
            pos, data_kwargs, meta_kwargs, out_indices = self._reconstruct_from_capture(cap)
            # init tensors
            for x in pos:
                if torch.is_tensor(x):
                    if fill == "randn": x.normal_()
                    elif fill == "ones": x.fill_(1)
                    elif fill == "zeros": x.zero_()
            grid = cap.resolved_grid if cap.resolved_grid is not None else (cap.grid_obj or 1)
            self._jit_func[grid](*pos, **data_kwargs, **meta_kwargs)
            _sync_all()
            return self._return_outputs(pos, out_indices)

        # Fallback to IOContract-based materialization
        concrete_args, out_indices = self._build_concrete_args_from_specs(())
        for x in concrete_args:
            if torch.is_tensor(x):
                if fill == "randn": x.normal_()
                elif fill == "ones": x.fill_(1)
                elif fill == "zeros": x.zero_()
        grid = self._resolve_triton_grid(self.kernel_code.io)
        # IO path only has meta from _build_args_from_io
        pos_args, ctx, meta, out_idx = self._build_args_from_io(self.kernel_code.io, ())
        self._jit_func[grid](*pos_args, **meta)
        _sync_all()
        return self._return_outputs(pos_args, out_idx)


class TritonCompilationBackend(BaseCompilationBackend):
    """Compilation strategy for Triton kernels"""
    
    def __init__(self, cache_dir: str = "/tmp/triton_cache"):
        self.cache_dir = cache_dir
        self.extractor = TritonKernelExtractor()
        self.logger = logger
        
    def compile(self, kernel: KernelCode, gpu_id: int, **kwargs) -> BaseExecutableKernel:
        """
        Compile Triton kernel into executable form
        
        Args:
            kernel: KernelCode with Triton source
            gpu_id: GPU device ID
            
        Returns:
            TritonExecutableKernel ready for execution
        """
        device = torch.device(f"cuda:{gpu_id}")
        # Validate kernel type
        if kernel.kernel_type != KernelType.TRITON:
            raise ValueError(f"Expected TRITON kernel type, got {kernel.kernel_type}")
        
        io = kernel.io
        # Extract kernel information
        self.logger.info("Extracting Triton kernel information")
        extracted_kernel_info: TritonKernelInfo = self.extractor.extract(kernel.source_code)
        # If source code contains kernel definition & invokes the kernel with a zero-arg function
        src_code_has_zero_arg_invoke = extracted_kernel_info.forward_function and not extracted_kernel_info.forward_function.get('has_input_args')
        # Build executable Triton kernel
        trit = TritonExecutableKernel(kernel, device=device, kernel_name=extracted_kernel_info.name)

        # If IOConctract not provided, then attempt to capture real launch inputs
        if io is None and src_code_has_zero_arg_invoke:
            try:
                io_contract = trit.capture_once(lambda: trit._loader.ns[extracted_kernel_info.forward_function.get('func_name')]())
                logger.info(f"Captured inputs: {[a.to_dict() for a in io_contract.args]}")
                _ = trit.compile_and_run()
            except Exception as e:
                self.logger.warning(f"Unable to auto-detect Triton kernel invocation: {e}")
                raise ValueError(f"Kernel does not contain invocation or input args! Kernel: {extracted_kernel_info}")

        # 2) Compilation/Execution check with synthetic input
        return trit    

    def cleanup(self):
        """Cleanup resources"""
        if self._loader:
            self._loader.cleanup()
            
    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass
