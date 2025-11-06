from typing import Optional
from KernelBench.scripts.cuda_eval_server_v2.io_contract import IOContractManager

from KernelBench.scripts.cuda_eval_server_v2.compilation.triton import TritonCompilationBackend
from KernelBench.scripts.cuda_eval_server_v2.shared.models import KernelCode, KernelType, IOContract, LaunchConfig, LaunchDim, TritonKernelMetadata
from validation.correctness_validator import CorrectnessValidator, ExecutableValidator
from profiling.kernel_profiler import ProfilingService
import torch
import triton

backend = TritonCompilationBackend()
ref_src = """
import triton 
import triton.language as tl
import torch
from typing import Optional

@triton.jit
def do_shrink_kernel(
    pid_n,
    pid_sk,
    slice_id,
    lora_index,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    ram,
    # input strides
    input_d0_stride,
    input_d1_stride,
    # lora strides
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    # output strides
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SLICE_NUM: tl.constexpr,
):
    \"\"\"
    Given an array of integers that identifies the rows of A, ram,
    a lora index that identifies which LoRA to use from lora_ptr, lora_index,
    a slice_id that identifies the input/output slice, compute the
    matrix product and store in the appropriate output location.
    \"\"\"

    # Identify the lora_ptr from slice_id.
    if SLICE_NUM == 1:
        # current lora ptr
        cur_lora_ptr = lora_ptr
    else:
        # current lora ptr
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(input_ptr.dtype.element_ty))

    # Identify the column indices of B to process.
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    # Identify A and B block pointers
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptr = (input_ptr + ram[:, None] * input_d0_stride +
             offset_k[None, :] * input_d1_stride)
    b_ptr = (cur_lora_ptr + lora_d0_stride * lora_index +
             rbn[None, :] * lora_d1_stride +
             offset_k[:, None] * lora_d2_stride)

    # Compute partial/complete block matrix product.
    accumulator = mm_k(a_ptr, b_ptr, input_d1_stride, lora_d2_stride, offset_k,
                       K, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, SPLIT_K, False,
                       cur_lora_ptr.dtype.element_ty)

    # Identify the C output pointers to store the results of the accumulator.
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_cm = tl.arange(0, BLOCK_M)
    cur_out_ptr = (out_ptr if SLICE_NUM == 1 else out_ptr +
                   slice_id * output_d0_stride)
    c_ptr = cur_out_ptr + ram[:, None] * output_d1_stride + offset_cn[
        None, :] * output_d2_stride
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < N)

    accumulator *= scaling
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask)


@triton.jit
def mm_k(a_ptr, b_ptr, ak_stride, bk_stride, offset_k, K: tl.constexpr,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
         EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr, CAST_TYPE: tl.constexpr,
         b_dtype: tl.constexpr):
    \"\"\"
    Given a_ptr and b_ptr, that identify the rows of A (m x k) and columns of
    B (k x n), iterate, through the K dimension to compute the partial/complete
    matrix block product.
    If SPLIT_K == 1, the output m x n product is complete.
    If SPLIT_K > 1, the thread block computes partial outputs. The partial
    outputs are then atomically summed in the caller code. 
    Args:
        a_ptr: Array of pointers, identifying rows of A 
        b_ptr: Array of pointers, identifying columns of B
        ak_stride: K dimension stride of the A matrix
        bk_stride: K dimension stride of the B matrix
        K: Length of the K dimension
        BLOCK_M: M dimension of the output block m x n
        BLOCK_N: N dimension of the output block m x n
        BLOCK_K: K dimension atom
        EVEN_K: True if the blocks of A and B can be loaded without any
          masking.
        SPLIT_K: Parameter signifying parallelism in the K dimension. 
        CAST_TYPE: if True, cast the values from the A matrix to the B
          matrix dtype.
        b_dtype: datatype of the B matrix
    \"\"\"
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            # A: random-gather stream → bypass L1, prefer not to keep
            tiled_a = tl.load(a_ptr, cache_modifier=".cg", eviction_policy="evict_first")
            # B: reused within CTA across K-steps → cache at all levels
            tiled_b = tl.load(b_ptr, cache_modifier=".ca", eviction_policy="evict_last")
        else:
            tiled_a = tl.load(
                a_ptr,
                mask=offset_k[None, :] < K - k * (BLOCK_K * SPLIT_K),
                other=0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )
            tiled_b = tl.load(
                b_ptr,
                mask=offset_k[:, None] < K - k * (BLOCK_K * SPLIT_K),
                other=0,
                cache_modifier=".ca",
                eviction_policy="evict_last",
            )
        if CAST_TYPE:
            tiled_a = tiled_a.to(b_dtype)
        accumulator += tl.dot(
            tiled_a,
            tiled_b,
        )
        a_ptr += BLOCK_K * SPLIT_K * ak_stride
        b_ptr += BLOCK_K * SPLIT_K * bk_stride
    return accumulator


@triton.jit
def _lora_shrink_kernel(input_ptr, lora_ptr, out_ptr, M, N, K,
                        token_indices_sorted_by_lora_ids, num_tokens_per_lora,
                        lora_token_start_loc, lora_ids, scaling,
                        input_d0_stride, input_d1_stride, lora_d0_stride,
                        lora_d1_stride, lora_d2_stride, output_d0_stride,
                        output_d1_stride, output_d2_stride,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr,
                        SPLIT_K: tl.constexpr, SLICE_NUM: tl.constexpr):

    # Calculate the number of thread blocks needed for N dimension (columns)
    cta_n_num = tl.cdiv(N, BLOCK_N)
    # Calculate the number of thread blocks needed for M dimension (rows/tokens)
    cta_m_num = tl.cdiv(M, BLOCK_M)

    # Get the current thread block's program ID from axis 0 (flattened SK, M, N dimensions)
    pid_sk_m_n = tl.program_id(axis=0)
    # Extract the split-K index from the flattened program ID
    pid_sk = pid_sk_m_n % SPLIT_K
    # Extract the M (row) block index from the flattened program ID
    pid_m = (pid_sk_m_n // SPLIT_K) % cta_m_num
    # Extract the N (column) block index from the flattened program ID
    pid_n = pid_sk_m_n // (SPLIT_K * cta_m_num) % cta_n_num

    # Get the slice ID from axis 1 (for handling multiple LoRA weight slices)
    slice_id = tl.program_id(axis=1)
    # Get the LoRA index from axis 2 (for handling multiple LoRAs in parallel)
    lora_idx = tl.program_id(axis=2)

    # Load the actual LoRA ID for this thread block from the lora_ids array
    lora_id = tl.load(lora_ids + lora_idx)
    # Check if this is a no-LoRA case (indicated by -1)
    if lora_id == -1:
        # Early exit for the no-lora case.
        return

    # Load the number of tokens that need to be processed by this LoRA
    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    # Calculate the starting row offset for this thread block
    cta_m_offset = pid_m * BLOCK_M
    # Check if this thread block is beyond the range of tokens for this LoRA
    if cta_m_offset >= lora_m_size:
        # Early exit CTA.
        return

    # Calculate the actual number of rows this thread block should process
    # (handles the case where the last block might be partial)
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)

    # Get the starting index in the sorted token indices array for this LoRA
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    # Calculate the pointer to the specific token indices this thread block should process
    cta_lora_seq_indices = (token_indices_sorted_by_lora_ids +
                            lora_m_indices_start + cta_m_offset)

    # Create offset array for loading token indices (wraps around for partial blocks)
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    # Load the actual row indices (token indices) that this thread block will operate on
    ram = tl.load(cta_lora_seq_indices + offset_m)

    # Call the actual computation kernel with all necessary parameters
    do_shrink_kernel(
        pid_n,                    # Column block index
        pid_sk,                   # Split-K index for parallel K-dimension processing
        slice_id,                 # Slice ID for multi-slice LoRA weights
        lora_id,                  # The actual LoRA ID to use
        input_ptr,                # Pointer to input tensor
        lora_ptr,                 # Pointer to LoRA weights
        out_ptr,                  # Pointer to output tensor
        N,                        # Number of columns (LoRA rank)
        K,                        # Number of input features (hidden size)
        cta_m_len,                # Number of rows this thread block processes
        ram,                      # Array of row indices to operate on
        # input strides
        input_d0_stride,          # Stride for input tensor dimension 0
        input_d1_stride,          # Stride for input tensor dimension 1
        # lora strides
        lora_d0_stride,           # Stride for LoRA tensor dimension 0
        lora_d1_stride,           # Stride for LoRA tensor dimension 1
        lora_d2_stride,           # Stride for LoRA tensor dimension 2
        # output strides
        output_d0_stride,         # Stride for output tensor dimension 0
        output_d1_stride,         # Stride for output tensor dimension 1
        output_d2_stride,         # Stride for output tensor dimension 2
        scaling,                  # Scaling factor to apply to results
        BLOCK_M,                  # Block size for M dimension
        BLOCK_N,                  # Block size for N dimension
        BLOCK_K,                  # Block size for K dimension
        EVEN_K,                   # Whether K dimension is evenly divisible
        SPLIT_K,                  # Split-K parallelization factor
        SLICE_NUM)                # Number of LoRA weight slices
"""


if __name__ == "__main__":
    _LORA_A_PTR_DICT: dict[tuple[int, ...], tuple[torch.tensor, ...]] = {}

    def _get_lora_a_ptr(lora_a_weights: list[torch.Tensor], device: torch.device):
        key = tuple(lora_weight.data_ptr() for lora_weight in lora_a_weights)

        if values := _LORA_A_PTR_DICT.get(key):
            return values

        lora_strides_d0 = []
        lora_strides_d1 = []
        lora_strides_d2 = []
        tensor_ptrs = []
        for lora_a_weight in lora_a_weights:
            if lora_a_weight.ndim == 4:  # shape:(lora_num,1,size,rank)
                assert lora_a_weight.size(1) == 1
                lora_a_weight = lora_a_weight.squeeze(dim=1)
            else:
                assert lora_a_weight.ndim == 3  # shape:(lora_num,size,rank)
            assert lora_a_weight.is_contiguous()
            tensor_ptrs.append(lora_a_weight.data_ptr())
            lora_strides_d0.append(lora_a_weight.stride(0))
            lora_strides_d1.append(lora_a_weight.stride(1))
            lora_strides_d2.append(lora_a_weight.stride(2))
        if len(lora_a_weights) > 1:
            lora_ptr_tensor = torch.tensor(tensor_ptrs,
                                        device=device,
                                        dtype=torch.uint64)
        else:
            lora_ptr_tensor = lora_a_weights[0]

        if (len(set(lora_strides_d0)) > 1 or len(set(lora_strides_d1)) > 1
                or len(set(lora_strides_d2)) > 1):
            raise ValueError("All LoRA weights must have the same stride.")

        _LORA_A_PTR_DICT[key] = (
            lora_ptr_tensor,
            lora_strides_d0[0],
            lora_strides_d1[0],
            lora_strides_d2[0],
        )
        return _LORA_A_PTR_DICT.get(key)

    @torch.inference_mode()
    def _lora_shrink(
        inputs: torch.Tensor,  #  shape [num_tokens, hidden_size]
        lora_a_weights: list[torch.Tensor],  # shape [num_loras, lora_rank, hidden_size]
        output_tensor: torch.Tensor,  # shape [num_slices, num_tokens, lora_rank]
        token_lora_mapping: torch.Tensor,  # shape [num_tokens]
        token_indices_sorted_by_lora_ids: torch.Tensor,  # shape [num_tokens] 
        num_tokens_per_lora: torch.Tensor,  # shape [max-loras + 1]
        lora_token_start_loc: torch.Tensor,  # shape [max-loras + 2]
        lora_ids: torch.Tensor,  # shape [max-loras + 1]
        scaling: float,
        kernel_config: Optional[dict] = None
    ) -> None:

        # breakpoint()
        assert inputs.dtype == lora_a_weights[0].dtype
        assert inputs.dtype in [torch.float16, torch.bfloat16]
        for weight in lora_a_weights:
            assert weight.dtype in [torch.float16, torch.bfloat16]

        assert inputs.size(1) == lora_a_weights[0].size(-1)
        assert inputs.is_contiguous()
        assert output_tensor.is_contiguous()

        # metadata sanity check
        M = inputs.size(0)
        assert token_lora_mapping.size(0) == M
        assert token_lora_mapping.size(0) == token_indices_sorted_by_lora_ids.size(0)
        assert lora_ids.size(0) == num_tokens_per_lora.size(0)
        assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1

        (lora_ptr_tensor, lora_strides_d0, lora_strides_d1,
        lora_strides_d2) = _get_lora_a_ptr(lora_a_weights, inputs.device)
        N, K = lora_a_weights[0].shape[-2:]  # K=hidden_size,N=rank
        NUM_SLICES = len(lora_a_weights)
        MAX_LORAS = lora_ids.size(0)
        if not kernel_config:
            kernel_config = {
                'block_m': 32,
                'block_n': 16,
                'block_k': 256 ,
                'split_k': 64 ,
                'num_warps': 4,
                'num_stages': 2,
            }
        BLOCK_M = kernel_config['block_m']
        BLOCK_N = kernel_config['block_n']
        BLOCK_K = kernel_config['block_k']
        SPLIT_K = kernel_config['split_k']
        NUM_WARPS = kernel_config['num_warps']
        NUM_STAGES = kernel_config['num_stages']

        EVEN_K = K % (BLOCK_K * SPLIT_K) == 0  # type: ignore

        # TODO (varun): This grid formulation maximizes parallelization at the
        # cost of wasteful thread block launch when only few of the input tokens
        # require LoRA. This might not be the best in all cases.
        grid = (
            SPLIT_K * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            NUM_SLICES,
            # Each LoRA receives its own set of thread blocks for output
            # computation. If some LoRA doesn't have any tokens to process, its
            # thread blocks exit early.
            MAX_LORAS,
        )

        #### --------------------------------

        io_manager = IOContractManager()

        ref_io = io_manager.create_io_contract(
            inputs=[
                inputs,
                lora_ptr_tensor,
                output_tensor,
                M,
                N,
                K,
                token_indices_sorted_by_lora_ids,
                num_tokens_per_lora,
                lora_token_start_loc,
                lora_ids,
                scaling,
                inputs.stride(0),
                inputs.stride(1),
                lora_strides_d0,
                lora_strides_d1,
                lora_strides_d2,
                output_tensor.stride(0),
                output_tensor.stride(1),
                output_tensor.stride(2),
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                EVEN_K,
                SPLIT_K,
                NUM_SLICES,
            ],
            output_indices=[2],
            launch_config=LaunchConfig(grid=LaunchDim(grid[0], grid[1], grid[2]), num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        )   

        ref_code = KernelCode(
            source_code=ref_src,
            kernel_type=KernelType.TRITON,
            io=ref_io,
            metadata=TritonKernelMetadata(kernel_name='_lora_shrink_kernel')
        )

        BLOCK_M = 64
        BLOCK_N = 32
        BLOCK_K = 64
        SPLIT_K = 4
        NUM_WARPS = 4
        NUM_STAGES = 3
        grid = (
            SPLIT_K * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            NUM_SLICES,
            # Each LoRA receives its own set of thread blocks for output
            # computation. If some LoRA doesn't have any tokens to process, its
            # thread blocks exit early.
            MAX_LORAS,
        )

        cand_io = io_manager.create_io_contract(
            inputs=[
                inputs,
                lora_ptr_tensor,
                output_tensor,
                M,
                N,
                K,
                token_indices_sorted_by_lora_ids,
                num_tokens_per_lora,
                lora_token_start_loc,
                lora_ids,
                scaling,
                inputs.stride(0),
                inputs.stride(1),
                lora_strides_d0,
                lora_strides_d1,
                lora_strides_d2,
                output_tensor.stride(0),
                output_tensor.stride(1),
                output_tensor.stride(2),
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                EVEN_K,
                SPLIT_K,
                NUM_SLICES,
            ],
            output_indices=[2],
            launch_config=LaunchConfig(grid=LaunchDim(grid[0], grid[1], grid[2]), num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        )   

        cand_code = KernelCode(
            source_code=ref_src,
            kernel_type=KernelType.TRITON,
            io=cand_io,
            metadata=TritonKernelMetadata(kernel_name='_lora_shrink_kernel')
        )
            
        ref_kernel = backend.compile(ref_code, gpu_id=0)
        cand_kernel = backend.compile(cand_code, gpu_id=0)
        validator = CorrectnessValidator()

        val = validator.validate_correctness(ref_kernel, cand_kernel, torch.device('cuda:0'), 2, 'abc', atol=1.0)
        print(f"{val=}")

        profiler = ProfilingService()
                
        # Create CompareProfilingResult object
        profiling_result = profiler.compare_profile(
            ref_kernel=ref_kernel,
            custom_kernel=cand_kernel,
            job_id='abc',
            gpu_id=0
        )
        print(f'{profiling_result=}')

    
    ## -------------------- START ---------------------------
    torch.manual_seed(0)
    
    num_tokens: int = 4096
    hidden_size: int = 2048
    lora_rank: int = 32
    num_loras: int = 16
    num_active_loras: int = 16
    num_slices: int = 1
    scaling: float = 1.0
    # Define missing variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    kernel_config = {
    'block_m': 32,
    'block_n': 16,
    'block_k': 256,
    'split_k': 64,
    'num_warps': 4,
    'num_stages': 2,
    }
    
    
    # Input tensor: [num_tokens, hidden_size]
    inputs = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    
    # LoRA A weights: [num_loras, lora_rank, hidden_size] 
    lora_a_weights = []
    for _ in range(num_slices):
        weight = torch.randn(num_loras, lora_rank, hidden_size, dtype=dtype, device=device)
        lora_a_weights.append(weight)
    
    # Output tensor: [num_slices, num_tokens, lora_rank]
    output_tensor = torch.zeros(num_slices, num_tokens, lora_rank, dtype=dtype, device=device)
    
    # Create token-to-LoRA mapping (some tokens use LoRA, others don't)
    token_lora_mapping = torch.randint(-1, num_active_loras, (num_tokens,), device=device)

    num_tokens = token_lora_mapping.size(0)
    
    # Analyze token distribution
    
    # Sort token indices by LoRA IDs
    _, token_indices_sorted_by_lora_ids = torch.sort(token_lora_mapping, stable=True)
    
    # Get unique LoRA IDs and their counts
    active_lora_ids, num_tokens_per_lora = torch.unique(
        token_lora_mapping, sorted=True, return_counts=True
    )
    
    # Pad the tensors to match expected sizes
    max_loras = num_loras
    
    # Pad num_tokens_per_lora to size max_loras
    padded_num_tokens_per_lora = torch.zeros(max_loras, dtype=torch.int32, device=device)
    actual_num_loras = min(len(num_tokens_per_lora), max_loras)
    if actual_num_loras > 0:
        padded_num_tokens_per_lora[:actual_num_loras] = num_tokens_per_lora[:actual_num_loras]
    
    # Pad active_lora_ids to size max_loras
    padded_lora_ids = torch.full((max_loras,), -1, dtype=torch.int32, device=device)
    if actual_num_loras > 0:
        padded_lora_ids[:actual_num_loras] = active_lora_ids[:actual_num_loras]
    
    # Calculate starting locations for each LoRA's tokens (size should be max_loras + 1)
    lora_token_start_loc = torch.zeros(max_loras + 1, dtype=torch.int32, device=device)
    if actual_num_loras > 0:
        lora_token_start_loc[1:actual_num_loras + 1] = torch.cumsum(padded_num_tokens_per_lora[:actual_num_loras], dim=0)

+        # --- NEW (R2): compact away -1 and empty LoRAs so we don't launch idle CTAs
+        active_mask = (padded_lora_ids != -1) & (padded_num_tokens_per_lora > 0)
+        lora_ids_compact = padded_lora_ids[active_mask]
+        num_tokens_per_lora_compact = padded_num_tokens_per_lora[active_mask]
+        # absolute starts into token_indices_sorted_by_lora_ids
+        lora_token_start_loc_compact = lora_token_start_loc[:-1][active_mask]
+        num_active_loras = int(active_mask.sum().item())

    # Calculate grid dimensions and analyze process distribution
    BLOCK_M = kernel_config['block_m']
    BLOCK_N = kernel_config['block_n']
    SPLIT_K = kernel_config['split_k']
    
    grid_dim_0 = SPLIT_K * triton.cdiv(num_tokens, BLOCK_M) * triton.cdiv(lora_rank, BLOCK_N)
    grid_dim_1 = num_slices
    grid_dim_2 = num_active_loras
    
    total_processes = grid_dim_0 * grid_dim_1 * grid_dim_2

    
    # Count processes that will exit due to lora_id == -
    no_lora_processes = 0
    active_processes = 0
    
    for lora_idx in range(max_loras):
        lora_id = padded_lora_ids[lora_idx].item()
        processes_for_this_lora = grid_dim_0 * grid_dim_1
        
        if lora_id == -1:
            no_lora_processes += processes_for_this_lora
    
        else:
            # Further analyze based on token count
            num_tokens_for_lora = padded_num_tokens_per_lora[lora_idx].item()
            blocks_needed = triton.cdiv(num_tokens_for_lora, BLOCK_M)
            total_blocks_allocated = triton.cdiv(num_tokens, BLOCK_M)
            
            # Processes that will do work
            working_processes = min(blocks_needed * SPLIT_K * triton.cdiv(lora_rank, BLOCK_N) * num_slices, processes_for_this_lora)
            # Processes that will exit due to cta_m_offset >= lora_m_size
            early_exit_processes = processes_for_this_lora - working_processes
            
            active_processes += working_processes
            no_lora_processes += early_exit_processes
    
    efficiency = (active_processes / total_processes) * 100 if total_processes > 0 else 0

    test_data = {
        "inputs": inputs,
        "lora_a_weights": lora_a_weights,
        "output_tensor": output_tensor,
        "token_lora_mapping": token_lora_mapping,
        "token_indices_sorted_by_lora_ids": token_indices_sorted_by_lora_ids,
        "num_tokens_per_lora": padded_num_tokens_per_lora,
        "lora_token_start_loc": lora_token_start_loc,
        "active_lora_ids": padded_lora_ids,
    }

    _lora_shrink(
        inputs=test_data["inputs"],
        lora_a_weights=test_data["lora_a_weights"],
        output_tensor=test_data["output_tensor"],
        token_lora_mapping=test_data["token_lora_mapping"],
        token_indices_sorted_by_lora_ids=test_data["token_indices_sorted_by_lora_ids"],
        num_tokens_per_lora=test_data["num_tokens_per_lora"],
        lora_token_start_loc=test_data["lora_token_start_loc"],
        lora_ids=test_data["active_lora_ids"],
        scaling=scaling,
        kernel_config=kernel_config
    )