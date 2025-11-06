from KernelBench.scripts.cuda_eval_server_v2.io_contract import IOContractManager

from KernelBench.scripts.cuda_eval_server_v2.compilation.triton import TritonCompilationBackend
from KernelBench.scripts.cuda_eval_server_v2.shared.models import KernelCode, KernelType, IOContract, LaunchConfig, LaunchDim
from validation.correctness_validator import ExecutableValidator
import torch

backend = TritonCompilationBackend()

#src = """
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import triton
import triton.language as tl

@triton.jit
def write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token,
                          token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N,
                          compute_type):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
                              offs_token, token_mask, BLOCK_SIZE_M,
                              BLOCK_SIZE_N, compute_type)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)
    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                            offs_bsn * stride_bsn)
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)
    if HAS_BIAS:
        # bias shape: [num_experts, N]
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                  mask=token_mask,
                                  other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if HAS_BIAS:
        accumulator = accumulator + bias[None, :]
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# """

def main():
    io_manager = IOContractManager()
    dtype = torch.float16
    # Default profiling dimensions
    top_k = 2
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128
    GROUP_SIZE_M = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    free_b = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
    target = int(max(2 << 30, min(8 << 30, free_b * 0.20))) if free_b else 1 << 20
    E, N, K = 16, 4096, 4096
    bytes_per_token = (K + top_k * N) * 2
    B_bytes = E * N * K * 2
    M = max(BLOCK_SIZE_M, (target - B_bytes) // bytes_per_token)
    M = int((M // BLOCK_SIZE_M) * BLOCK_SIZE_M)
    if (not torch.cuda.is_available()) or M < BLOCK_SIZE_M:
        # SMALL_SANITY fallback when CUDA is unavailable or memory is low
        M, E, N, K, top_k = 128, 2, 64, 64, 1
        BLOCK_SIZE_M = BLOCK_SIZE_N = BLOCK_SIZE_K = 64
        GROUP_SIZE_M = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(E, N, K, device=device, dtype=dtype)
    C = torch.empty(M, top_k, N, device=device, dtype=dtype)
    sorted_token_ids = torch.arange(M * top_k, device=device, dtype=torch.int32)
    expert_ids = torch.arange((M * top_k) // BLOCK_SIZE_M, device=device, dtype=torch.int32) % E
    num_tokens_post_padded = torch.tensor([M * top_k], device=device, dtype=torch.int32)

    grid = (triton.cdiv(M * top_k, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    fused_moe_kernel[grid](
        A, B, C, None, None, None, None,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, M * top_k, M * top_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(1), C.stride(2),
        0, 0, 0, 0, 0, 0, 0,
        0, 0,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        MUL_ROUTED_WEIGHT=False,
        top_k=top_k,
        compute_type=tl.float16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        HAS_BIAS=False,
    )


    # grid = LaunchDim(num_seqs, num_kv_heads)
    # io = io_manager.create_io_contract([
    #     output, query, key_cache, value_cache, empty, block_tables, seq_lens, empty,
    #     1.0, k_scale, v_scale,
    #     num_query_heads,
    #     num_queries_per_kv,
    #     num_queries_per_kv_padded,
    #     block_tables.stride(0),
    #     query.stride(0),
    #     query.stride(1),
    #     output.stride(0),
    #     output.stride(1),
    #     block_size,
    #     head_size,
    #     head_size_padded,
    #     False,
    #     0,
    #     x,
    #     key_cache.stride(0),
    #     key_cache.stride(1),
    #     key_cache.stride(2),
    #     key_cache.stride(3),
    #     key_cache.stride(4),
    #     value_cache.stride(0),
    #     value_cache.stride(1),
    #     value_cache.stride(2),
    #     value_cache.stride(3),
    #     False,
    #     empty_int,
    #     False],
    #     [0],
    #     launch_config=LaunchConfig(grid=grid)
    # )

    # io = IOContract.from_dict({
    #     "args": [
    #         {
    #         "name": "output_ptr",
    #         "type": "tensor",
    #         "role": "output",
    #         "tensor_spec": {
    #             "shape": [8, 32, 128],
    #             "dtype": "float16"
    #         }
    #         },
    #         {
    #         "name": "query_ptr",
    #         "type": "tensor",
    #         "role": "input",
    #         "tensor_spec": {
    #             "shape": [8, 32, 128],
    #             "dtype": "float16",
    #             "init": {"kind": "randn", "seed": 1}
    #         }
    #         },
    #         {
    #         "name": "key_cache_ptr",
    #         "type": "tensor",
    #         "role": "input",
    #         "tensor_spec": {
    #             "shape": [512, 8, 16, 16, 8],
    #             "dtype": "float16",
    #             "init": {"kind": "randn", "seed": 2}
    #         }
    #         },
    #         {
    #         "name": "value_cache_ptr",
    #         "type": "tensor",
    #         "role": "input",
    #         "tensor_spec": {
    #             "shape": [512, 8, 128, 16],
    #             "dtype": "float16",
    #             "init": {"kind": "randn", "seed": 3}
    #         }
    #         },
    #         {
    #         "name": "sink_ptr",
    #         "type": "tensor",
    #         "role": "input",
    #         "tensor_spec": {
    #             "shape": [32],
    #             "dtype": "float32",
    #             "init": {"kind": "zeros"}
    #         }
    #         },
    #         {
    #         "name": "block_tables_ptr",
    #         "type": "tensor",
    #         "role": "input",
    #         "tensor_spec": {
    #             "shape": [8, 64],
    #             "dtype": "int32",
    #             "init": {"kind": "arange", "start": 0, "step": 1}
    #         }
    #         },
    #         {
    #         "name": "seq_lens_ptr",
    #         "type": "tensor",
    #         "role": "input",
    #         "tensor_spec": {
    #             "shape": [8],
    #             "dtype": "int32",
    #             "init": {"kind": "full", "fill_value": 256}
    #         }
    #         },
    #         {
    #         "name": "alibi_slopes_ptr",
    #         "type": "tensor",
    #         "role": "input",
    #         "tensor_spec": {
    #             "shape": [32],
    #             "dtype": "float32",
    #             "init": {"kind": "uniform", "low": -0.1, "high": 0.0, "seed": 4}
    #         }
    #         },
    #         {
    #         "name": "scale",
    #         "type": "float",
    #         "value": 0.08838834764831845,
    #         "role": "input"
    #         },
    #         {
    #         "name": "k_scale",
    #         "type": "float",
    #         "value": 1.0,
    #         "role": "input"
    #         },
    #         {
    #         "name": "v_scale",
    #         "type": "float",
    #         "value": 1.0,
    #         "role": "input"
    #         },
    #         {
    #         "name": "num_query_heads",
    #         "type": "int",
    #         "value": 32,
    #         "is_meta": True
    #         },
    #         {
    #         "name": "num_queries_per_kv",
    #         "type": "int",
    #         "value": 4,
    #         "is_meta": True
    #         },
    #         {
    #         "name": "num_queries_per_kv_padded",
    #         "type": "int",
    #         "value": 4,
    #         "is_meta": True
    #         },
    #         {
    #         "name": "block_table_stride",
    #         "type": "int",
    #         "value": 64,
    #         "role": "input"
    #         },
    #         {
    #         "name": "query_stride_0",
    #         "type": "int",
    #         "value": 4096,
    #         "role": "input"
    #         },
    #         {
    #         "name": "query_stride_1",
    #         "type": "int",
    #         "value": 128,
    #         "role": "input"
    #         },
    #         {
    #         "name": "output_stride_0",
    #         "type": "int",
    #         "value": 4096,
    #         "role": "input"
    #         },
    #         {
    #         "name": "output_stride_1",
    #         "type": "int",
    #         "value": 128,
    #         "role": "input"
    #         },
    #         {
    #         "name": "BLOCK_SIZE",
    #         "type": "int",
    #         "value": 16,
    #         "is_meta": True
    #         },
    #         {
    #         "name": "HEAD_SIZE",
    #         "type": "int",
    #         "value": 128,
    #         "is_meta": True
    #         },
    #         {
    #         "name": "HEAD_SIZE_PADDED",
    #         "type": "int",
    #         "value": 128,
    #         "is_meta": True
    #         },
    #         {
    #         "name": "USE_ALIBI_SLOPES",
    #         "type": "bool",
    #         "value": True,
    #         "is_meta": True
    #         },
    #         {
    #         "name": "SLIDING_WINDOW",
    #         "type": "int",
    #         "value": -1,
    #         "is_meta": True
    #         },
    #         {
    #         "name": "x",
    #         "type": "int",
    #         "value": 8,
    #         "is_meta": True
    #         },
    #         {
    #         "name": "stride_k_cache_0",
    #         "type": "int",
    #         "value": 2048,
    #         "role": "input"
    #         },
    #         {
    #         "name": "stride_k_cache_1",
    #         "type": "int",
    #         "value": 256,
    #         "role": "input"
    #         },
    #         {
    #         "name": "stride_k_cache_2",
    #         "type": "int",
    #         "value": 128,
    #         "role": "input"
    #         },
    #         {
    #         "name": "stride_k_cache_3",
    #         "type": "int",
    #         "value": 8,
    #         "role": "input"
    #         },
    #         {
    #         "name": "stride_k_cache_4",
    #         "type": "int",
    #         "value": 1,
    #         "role": "input"
    #         },
    #         {
    #         "name": "stride_v_cache_0",
    #         "type": "int",
    #         "value": 16384,
    #         "role": "input"
    #         },
    #         {
    #         "name": "stride_v_cache_1",
    #         "type": "int",
    #         "value": 2048,
    #         "role": "input"
    #         },
    #         {
    #         "name": "stride_v_cache_2",
    #         "type": "int",
    #         "value": 16,
    #         "role": "input"
    #         },
    #         {
    #         "name": "stride_v_cache_3",
    #         "type": "int",
    #         "value": 1,
    #         "role": "input"
    #         },
    #         {
    #         "name": "filter_by_query_len",
    #         "type": "bool",
    #         "value": False,
    #         "is_meta": True
    #         },
    #         {
    #         "name": "query_start_len_ptr",
    #         "type": "tensor",
    #         "role": "input",
    #         "tensor_spec": {
    #             "shape": [9],
    #             "dtype": "int32",
    #             "init": {"kind": "arange", "start": 0, "step": 1}
    #         }
    #         },
    #         {
    #         "name": "USE_SINKS",
    #         "type": "bool",
    #         "value": False,
    #         "is_meta": True
    #         }
    #     ],
    #     "launch": {
    #         "grid": {"x": 8, "y": 8},
    #         "num_warps": 4,
    #         "num_stages": 2
    #     }
    # }
    # )

    # kernel_code = KernelCode(
    #     source_code=src,
    #     kernel_type=KernelType.TRITON,
    #     io=io
    # )
        
    #executable = backend.compile(kernel_code, gpu_id=0)
    # validator = ExecutableValidator()

    # result = validator.validate(executable, torch.device('cuda:0'), 2, 'abc')
    result = C
    print(f"{result.min()}")

main()