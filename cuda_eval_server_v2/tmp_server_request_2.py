from typing import Optional
from io_contract import IOContractManager

from KernelBench.scripts.cuda_eval_server_v2.compilation.triton import TritonCompilationBackend
from KernelBench.scripts.cuda_eval_server_v2.shared.models import KernelCode, KernelType, IOContract, LaunchConfig, LaunchDim, TritonKernelMetadata
import torch
import triton

backend = TritonCompilationBackend()
"""
End-to-end test for custom tolerance feature
Tests that atol/rtol parameters flow through the API correctly
"""

import requests
import json

# Test configuration
SERVER_URL = "http://localhost:8000"

# Simple PyTorch reference kernel
ref_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def mm_k(a_ptr, b_ptr, ak_stride, bk_stride, offset_k, K: tl.constexpr,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
         EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr, CAST_TYPE: tl.constexpr,
         b_dtype: tl.constexpr):
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            tiled_b = tl.load(b_ptr)
        else:
            tiled_a = tl.load(a_ptr,
                              mask=offset_k[None, :]
                              < K - k * (BLOCK_K * SPLIT_K),
                              other=0)
            tiled_b = tl.load(b_ptr,
                              mask=offset_k[:, None]
                              < K - k * (BLOCK_K * SPLIT_K),
                              other=0)
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
def do_expand_kernel(
    pid_n,
    lora_index,
    slice_id,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    ram,  # array identifying the rows of Input ptr to operate on
    slice_start_loc,
    # input ptr strides
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,
    # lora ptr strides
    ls_d0_ptr,
    ls_d1_ptr,
    ls_d2_ptr,
    # out ptr strides
    output_d0_stride,
    output_d1_stride,
    # constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    EVEN_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
):
    # ls_d*_ptr can be either an integer or a pointer
    if SAME_STRIDE:
        # integer
        cur_lora_d0_stride = ls_d0_ptr
        cur_lora_d1_stride = ls_d1_ptr
        cur_lora_d2_stride = ls_d2_ptr
    else:
        # pointer
        cur_lora_d0_stride = tl.load(ls_d0_ptr + slice_id)
        cur_lora_d1_stride = tl.load(ls_d1_ptr + slice_id)
        cur_lora_d2_stride = tl.load(ls_d2_ptr + slice_id)

    # Identify the input_ptr and lora_ptr from slice_id.
    if SLICE_NUM == 1:
        cur_input_ptr = input_ptr
        cur_lora_ptr = lora_ptr
    else:
        cur_input_ptr = input_ptr + slice_id * input_d0_stride
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(out_ptr.dtype.element_ty))

    # Identify the column indices of B to process.
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    # Identify A and B block pointers
    offset_k = tl.arange(0, BLOCK_K)
    a_ptr = (cur_input_ptr + ram[:, None] * input_d1_stride +
             offset_k[None, :] * input_d2_stride)
    b_ptr = (cur_lora_ptr + cur_lora_d0_stride * lora_index +
             offset_k[:, None] * cur_lora_d2_stride +
             rbn[None, :] * cur_lora_d1_stride)

    # Compute the block matrix product.
    SPLIT_K = 1
    accumulator = mm_k(a_ptr, b_ptr, input_d2_stride, cur_lora_d2_stride,
                       offset_k, K, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, SPLIT_K,
                       CAST_TYPE, cur_lora_ptr.dtype.element_ty)

    tiled_c = accumulator.to(cur_lora_ptr.dtype.element_ty)
    if SLICE_NUM == 1:
        cur_slice_start = slice_start_loc
    else:
        cur_slice_start = tl.load(slice_start_loc + slice_id)

    # Identify the C output pointers to store the results of the accumulator.
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + cur_slice_start
    offset_cm = tl.arange(0, BLOCK_M)
    c_ptr = (out_ptr + ram[:, None] * output_d0_stride +
             offset_cn[None, :] * output_d1_stride)
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :]
                                             < (cur_slice_start + N))

    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)

@triton.jit
def _lora_expand_kernel(
        input_ptr,
        lora_ptr,
        out_ptr,
        M,
        N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        slice_start_loc,
        input_d0_stride,
        input_d1_stride,
        input_d2_stride,  # 1
        ls_d0_ptr,
        ls_d1_ptr,
        ls_d2_ptr,  # 1
        output_d0_stride,
        output_d1_stride,  # 1
        output_hs_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        ADD_INPUTS: tl.constexpr,
        CAST_TYPE: tl.constexpr,
        SLICE_NUM: tl.constexpr,
        SAME_STRIDE: tl.constexpr):

    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)

    pid_mn = tl.program_id(axis=0)
    pid_m = pid_mn % cta_m_num
    pid_n = (pid_mn // cta_m_num) % cta_n_num

    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        # Early exit for the no-lora case.
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        # Early exit CTA.
        return

    # When the output dimensions of each slice are the same,cur_n=N, otherwise
    # cur_n=tl.load(output_hs_ptr + slice_id), this situation exists in GQA's
    # qkv linear.
    curr_N = N if SAME_STRIDE else tl.load(output_hs_ptr + slice_id)
    if pid_n * BLOCK_N >= curr_N:
        # Early exit CTA.
        return

    # num rows this CTA should process.
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)

    # Identify all rows that this CTA should process.
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (token_indices_sorted_by_lora_ids +
                            lora_m_indices_start + cta_m_offset)

    # Load all relevant row indices.
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    do_expand_kernel(
        pid_n,
        lora_id,
        slice_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        curr_N,
        K,
        cta_m_len,
        ram,  # array identifying the rows of Input ptr to operate on
        slice_start_loc,
        # input ptr strides
        input_d0_stride,
        input_d1_stride,
        input_d2_stride,
        # lora ptr strides
        ls_d0_ptr,
        ls_d1_ptr,
        ls_d2_ptr,
        # out ptr strides
        output_d0_stride,
        output_d1_stride,
        # constants
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        SAME_STRIDE,
        SLICE_NUM,
        EVEN_K,
        CAST_TYPE,
        ADD_INPUTS)
"""

if __name__ == "__main__":
    _LORA_B_PTR_DICT: dict[tuple[int, ...], tuple[torch.tensor, ...]] = {}
    
    def _get_lora_b_ptr(lora_weights: list[torch.Tensor], offset_start: int,
                        device: torch.device):
        key = tuple(lora_weight.data_ptr() for lora_weight in lora_weights)
        if values := _LORA_B_PTR_DICT.get(key):
            return values
        slice_offset_lst = []
        tensor_ptrs = []
        lora_strides_d0 = []
        lora_strides_d1 = []
        lora_strides_d2 = []
        hidden_sizes = []
        slice_offset = offset_start
        for lora_b_weight in lora_weights:
            if lora_b_weight.ndim == 4:  # shape:(lora_num,1,size,rank)
                assert lora_b_weight.size(1) == 1
                lora_b_weight = lora_b_weight.squeeze(dim=1)
            else:
                assert lora_b_weight.ndim == 3  # shape:(lora_num,size,rank)
            assert lora_b_weight.is_contiguous()
            tensor_ptrs.append(lora_b_weight.data_ptr())
            lora_strides_d0.append(lora_b_weight.stride(0))
            lora_strides_d1.append(lora_b_weight.stride(1))
            lora_strides_d2.append(lora_b_weight.stride(2))
            slice_offset_lst.append(slice_offset)
            slice_offset += lora_b_weight.size(1)
            hidden_sizes.append(lora_b_weight.size(1))

        if len(lora_weights) > 1:
            # note these are device tensors
            lora_ptr_tensor = torch.tensor(tensor_ptrs,
                                        device=device,
                                        dtype=torch.uint64)
            slice_start_tensor = torch.tensor(slice_offset_lst,
                                            device=device,
                                            dtype=torch.uint64)
        else:
            slice_start_tensor = slice_offset_lst[0]
            lora_ptr_tensor = lora_b_weight[0]

        # If each lora has the same stride, there's no need to use a
        # tensor for storage.
        if (len(set(lora_strides_d0)) == 1 and len(set(lora_strides_d1)) == 1 and
                len(set(lora_strides_d2)) == 1) and len(set(hidden_sizes)) == 1:
            lora_strides_d0_tensor = lora_strides_d0[0]
            lora_strides_d1_tensor = lora_strides_d1[0]
            lora_strides_d2_tensor = lora_strides_d2[0]
            hidden_sizes_tensor = hidden_sizes[0]
            same_stride = True

        else:
            lora_strides_d0_tensor = torch.tensor(lora_strides_d0, device=device)
            lora_strides_d1_tensor = torch.tensor(lora_strides_d1, device=device)
            lora_strides_d2_tensor = torch.tensor(lora_strides_d2, device=device)
            hidden_sizes_tensor = torch.tensor(hidden_sizes, device=device)
            same_stride = False
        # MAX_N is the maximum hidden size among all the lora_b weights
        MAX_N = max(hidden_sizes)
        _LORA_B_PTR_DICT[key] = (slice_start_tensor, lora_ptr_tensor,
                                lora_strides_d0_tensor, lora_strides_d1_tensor,
                                lora_strides_d2_tensor, hidden_sizes_tensor,
                                same_stride, MAX_N)
        return _LORA_B_PTR_DICT.get(key)


    torch.manual_seed(0)

    num_tokens = 8192
    hidden_size = 5120
    lora_rank = 32
    num_loras = 16
    num_active_loras = 16
    num_slices = 1
    offset_start = 0
    add_inputs = False
    dtype = torch.float16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Input tensor: [num_slices, num_tokens, lora_rank]
    # Note: For expand, input dtype can be float32 while weights are float16/bfloat16
    inputs = torch.randn(num_slices, num_tokens, lora_rank, dtype=torch.float32, device=device)
    
    # LoRA B weights: [num_loras, hidden_size, lora_rank] 
    lora_b_weights = []
    for _ in range(num_slices):
        weight = torch.randn(num_loras, hidden_size, lora_rank, dtype=dtype, device=device)
        lora_b_weights.append(weight)
    
    # Output tensor: [num_tokens, hidden_size * num_slices]
    output_tensor = torch.zeros(num_tokens, hidden_size * num_slices, dtype=dtype, device=device)
    
    # Create token-to-LoRA mapping
    token_lora_mapping = torch.randint(-1, num_active_loras, (num_tokens,), device=device)
    
    # Sort token indices by LoRA IDs
    _, token_indices_sorted_by_lora_ids = torch.sort(token_lora_mapping, stable=True)
    
    # Get unique LoRA IDs and their counts
    active_lora_ids, num_tokens_per_lora_raw = torch.unique(
        token_lora_mapping, sorted=True, return_counts=True
    )
    
    # Pad the tensors to match expected sizes
    max_loras = num_loras
    
    # Pad num_tokens_per_lora to size max_loras + 1
    padded_num_tokens_per_lora = torch.zeros(max_loras + 1, dtype=torch.int32, device=device)
    actual_num_loras = min(len(num_tokens_per_lora_raw), max_loras)
    if actual_num_loras > 0:
        padded_num_tokens_per_lora[:actual_num_loras] = num_tokens_per_lora_raw[:actual_num_loras]
    
    # Pad active_lora_ids to size max_loras + 1
    padded_lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32, device=device)
    if actual_num_loras > 0:
        padded_lora_ids[:actual_num_loras] = active_lora_ids[:actual_num_loras]
    
    # Calculate starting locations for each LoRA's tokens (size should be max_loras + 2)
    lora_token_start_loc = torch.zeros(max_loras + 2, dtype=torch.int32, device=device)
    if actual_num_loras > 0:
        lora_token_start_loc[1:actual_num_loras + 1] = torch.cumsum(padded_num_tokens_per_lora[:actual_num_loras], dim=0)
    
    # Assertion checks from _lora_expand function
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    for weight in lora_b_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]
    
    assert inputs.size(0) == len(lora_b_weights)
    assert output_tensor.is_contiguous()
    
    # metadata sanity check
    M = inputs.size(1)
    assert token_lora_mapping.size(0) == M
    assert token_lora_mapping.size(0) == token_indices_sorted_by_lora_ids.size(0)
    assert padded_lora_ids.size(0) == padded_num_tokens_per_lora.size(0)
    assert lora_token_start_loc.size(0) == padded_lora_ids.size(0) + 1
    
    # Get LoRA B pointer and strides
    (slice_start_tensor, lora_ptr_tensor, lora_strides_d0_tensor,
     lora_strides_d1_tensor, lora_strides_d2_tensor, hidden_sizes_tensor,
     same_stride, MAX_N) = _get_lora_b_ptr(lora_b_weights, offset_start, device)
    
    # Extract dimensions
    K = lora_b_weights[0].shape[-1]  # K = rank
    M = inputs.size(1)  # num_tokens
    NUM_SLICES = len(lora_b_weights)
    MAX_LORAS = padded_lora_ids.size(0)
    
    # Extract kernel configuration
    BLOCK_M = 16
    BLOCK_N = 128
    BLOCK_K = 16
    NUM_WARPS = 2
    NUM_STAGES = 3
    
    # Calculate kernel constants
    EVEN_K = K % BLOCK_K == 0
    ADD_INPUTS = add_inputs
    CAST_TYPE = False
    
    # Check if we need type casting (float32 input with float16/bfloat16 weights)
    if inputs.dtype == torch.float32 and lora_b_weights[0].dtype in [torch.float16, torch.bfloat16]:
        CAST_TYPE = True
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(MAX_N, BLOCK_N),
        NUM_SLICES,
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
            MAX_N,
            K,
            token_indices_sorted_by_lora_ids,
            padded_num_tokens_per_lora,
            lora_token_start_loc,
            padded_lora_ids,
            slice_start_tensor,
            inputs.stride(0),
            inputs.stride(1),
            inputs.stride(2),
            lora_strides_d0_tensor,
            lora_strides_d1_tensor,
            lora_strides_d2_tensor,
            output_tensor.stride(0),
            output_tensor.stride(1),
            hidden_sizes_tensor,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            EVEN_K,
            ADD_INPUTS,
            CAST_TYPE,
            NUM_SLICES,
            same_stride,
        ],
        output_indices=[2],
        launch_config=LaunchConfig(grid=LaunchDim(grid[0], grid[1], grid[2]), num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    )

    request = {
        "kernel": {
            "source_code": ref_code,
            "kernel_type": "triton",
            "metadata": {
                "kernel_name": "_lora_expand_kernel",
            },
            "io": ref_io.to_dict()
        },
        "num_trials": 100
    }

    response = requests.post(f"{SERVER_URL}/evaluate", json=request)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    result = response.json()
    print(result) 
