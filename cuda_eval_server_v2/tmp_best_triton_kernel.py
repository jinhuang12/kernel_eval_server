from KernelBench.scripts.cuda_eval_server_v2.io_contract import IOContractManager

from KernelBench.scripts.cuda_eval_server_v2.compilation.triton import TritonCompilationBackend
from KernelBench.scripts.cuda_eval_server_v2.shared.models import KernelCode, KernelType, IOContract
from validation.correctness_validator import CorrectnessValidator
from profiling.kernel_profiler import ProfilingService
import torch

backend = TritonCompilationBackend()
ref_src = """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_paged_attention_2d(
        output_ptr,  # [num_tokens, num_query_heads, head_size]
        query_ptr,  # [num_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
        value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
        sink_ptr,  # [num_query_heads]
        block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
        seq_lens_ptr,  # [num_seqs]
        alibi_slopes_ptr,  # [num_query_heads]
        scale,  # float32
        k_scale,  # float32
        v_scale,  # float32
        num_query_heads: tl.constexpr,  # int
        num_queries_per_kv: tl.constexpr,  # int
        num_queries_per_kv_padded: tl.constexpr,  # int
        block_table_stride: tl.int64,  # int
        query_stride_0: tl.int64,  # int
        query_stride_1: tl.int64,  # int, should be equal to head_size
        output_stride_0: tl.int64,  # int
        output_stride_1: tl.int64,  # int, should be equal to head_size
        BLOCK_SIZE: tl.constexpr,  # int
        HEAD_SIZE: tl.constexpr,  # int
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        USE_ALIBI_SLOPES: tl.constexpr,  # bool
        SLIDING_WINDOW: tl.constexpr,  # int
        x: tl.constexpr,  # int
        stride_k_cache_0: tl.int64,  # int
        stride_k_cache_1: tl.int64,  # int
        stride_k_cache_2: tl.int64,  # int
        stride_k_cache_3: tl.int64,  # int
        stride_k_cache_4: tl.int64,  # int
        stride_v_cache_0: tl.int64,  # int
        stride_v_cache_1: tl.int64,  # int
        stride_v_cache_2: tl.int64,  # int
        stride_v_cache_3: tl.int64,  # int
        filter_by_query_len: tl.constexpr,  # bool
        query_start_len_ptr,  # [num_seqs+1]
        USE_SINKS: tl.constexpr,  # bool
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx +
                                              1)
        cur_batch_query_len = cur_batch_in_all_stop_index             - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded)

    query_offset = (cur_batch_in_all_start_index * query_stride_0 +
                    query_head_idx[:, None] * query_stride_1)

    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1,
                        0).to(tl.int1)

    # Q : (num_queries_per_kv, HEAD_SIZE,)
    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([num_queries_per_kv_padded],
                    float(\"-inf\"),
                    dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_head_idx,
            mask=head_mask,
            other=float(\"-inf\"),
        ).to(dtype=tl.float32)

    L = tl.full([num_queries_per_kv_padded], 1.0, dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED],
                   dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx,
                              mask=head_mask,
                              other=0.0)

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    # iterate through tiles
    for j in range(0, num_blocks):

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)

        v_offset = (physical_block_idx * stride_v_cache_0 +
                    kv_head_idx * stride_v_cache_1 +
                    offs_d[None, :] * stride_v_cache_2 +
                    offs_n[:, None] * stride_v_cache_3)

        k_offset = (physical_block_idx * stride_k_cache_0 +
                    kv_head_idx * stride_k_cache_1 +
                    (offs_d[:, None] // x) * stride_k_cache_2 +
                    offs_n[None, :] * stride_k_cache_3 +
                    (offs_d[:, None] % x) * stride_k_cache_4)

        # K : (HEAD_SIZE, BLOCK_SIZE)
        K_load = tl.load(key_cache_ptr + k_offset,
                         mask=dim_mask[:, None],
                         other=0.0)

        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (BLOCK_SIZE, HEAD_SIZE)
        V_load = tl.load(value_cache_ptr + v_offset,
                         mask=dim_mask[None, :],
                         other=0.0)

        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < boundary

        # S : (num_queries_per_kv, BLOCK_SIZE,)
        S = tl.where(head_mask[:, None] & seq_mask, 0.0,
                     float(\"-inf\")).to(tl.float32)
        S += scale * tl.dot(Q, K)

        context_len = seq_len - 1

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S,
                         -10000)

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        # compute running maximum
        # m_j : (num_queries_per_kv,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # P : (num_queries_per_kv, BLOCK_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (num_queries_per_kv,)
        l_j = tl.sum(P, axis=1)

        # alpha : (num_queries_per_kv, )
        alpha = tl.exp(M - m_j)

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]

    output_offset = (cur_batch_in_all_start_index * output_stride_0 +
                     query_head_idx * output_stride_1)

    tl.store(
        output_ptr + output_offset[:, None] +
        tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        acc,
        mask=dim_mask[None, :] & head_mask[:, None],
    )

@triton.jit
def cdiv_fn(x, y):
 return (x + y - 1) // y

"""


cand_src = """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_paged_attention_2d(
        output_ptr,  # [num_tokens, num_query_heads, head_size]
        query_ptr,  # [num_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
        value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
        sink_ptr,  # [num_query_heads]
        block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
        seq_lens_ptr,  # [num_seqs]
        alibi_slopes_ptr,  # [num_query_heads]
        scale,  # float32
        k_scale,  # float32
        v_scale,  # float32
        num_query_heads: tl.constexpr,  # int
        num_queries_per_kv: tl.constexpr,  # int
        num_queries_per_kv_padded: tl.constexpr,  # int
        block_table_stride: tl.int64,  # int
        query_stride_0: tl.int64,  # int
        query_stride_1: tl.int64,  # int, should be equal to head_size
        output_stride_0: tl.int64,  # int
        output_stride_1: tl.int64,  # int, should be equal to head_size
        BLOCK_SIZE: tl.constexpr,  # int
        HEAD_SIZE: tl.constexpr,  # int
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        USE_ALIBI_SLOPES: tl.constexpr,  # bool
        SLIDING_WINDOW: tl.constexpr,  # int
        x: tl.constexpr,  # int
        stride_k_cache_0: tl.int64,  # int
        stride_k_cache_1: tl.int64,  # int
        stride_k_cache_2: tl.int64,  # int
        stride_k_cache_3: tl.int64,  # int
        stride_k_cache_4: tl.int64,  # int
        stride_v_cache_0: tl.int64,  # int
        stride_v_cache_1: tl.int64,  # int
        stride_v_cache_2: tl.int64,  # int
        stride_v_cache_3: tl.int64,  # int
        filter_by_query_len: tl.constexpr,  # bool
        query_start_len_ptr,  # [num_seqs+1]
        USE_SINKS: tl.constexpr,  # bool
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    # Early exit for filter_by_query_len
    if filter_by_query_len:
        cur_batch_start = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_stop = tl.load(query_start_len_ptr + seq_idx + 1)
        if cur_batch_stop - cur_batch_start > 1:
            return
        cur_batch_idx = cur_batch_start
    else:
        cur_batch_idx = seq_idx

    # Compute query head indices
    query_head_start = kv_head_idx * num_queries_per_kv
    query_head_offs = tl.arange(0, num_queries_per_kv_padded)
    query_head_idx = query_head_start + query_head_offs
    head_mask = query_head_offs < num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    # Load query
    q_offs = cur_batch_idx * query_stride_0 + query_head_idx[:, None] * query_stride_1
    dim_offs = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = dim_offs < HEAD_SIZE
    
    Q = tl.load(
        query_ptr + q_offs + dim_offs[None, :],
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0
    )

    # Initialize accumulator state
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    
    if USE_SINKS:
        M = tl.load(sink_ptr + query_head_idx, mask=head_mask, other=float("-inf")).to(tl.float32)
    else:
        M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    
    L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    # Load alibi slopes if needed
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx, mask=head_mask, other=0.0)

    # Iterate over blocks
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_table_base = seq_idx * block_table_stride

    for block_idx in range(num_blocks):
        # Load block index
        physical_block = tl.load(block_tables_ptr + block_table_base + block_idx)
        
        # Compute key/value offsets
        token_offs = tl.arange(0, BLOCK_SIZE)
        
        # Key cache layout: [num_blks, num_kv_heads, head_size // x, blk_size, x]
        k_base = (physical_block * stride_k_cache_0 + 
                  kv_head_idx * stride_k_cache_1)
        
        # Value cache layout: [num_blks, num_kv_heads, head_size, blk_size]  
        v_base = (physical_block * stride_v_cache_0 + 
                  kv_head_idx * stride_v_cache_1)

        # Load keys - simplified layout handling
        k_offs = (k_base + 
                  (dim_offs[:, None] // x) * stride_k_cache_2 + 
                  token_offs[None, :] * stride_k_cache_3 +
                  (dim_offs[:, None] % x) * stride_k_cache_4)
        
        K = tl.load(
            key_cache_ptr + k_offs,
            mask=dim_mask[:, None],
            other=0.0
        )
        
        # Apply k_scale if needed
        if K.dtype.is_fp8():
            K = (K.to(tl.float32) * k_scale).to(Q.dtype)

        # Load values
        v_offs = (v_base + 
                  dim_offs[None, :] * stride_v_cache_2 + 
                  token_offs[:, None] * stride_v_cache_3)
        
        V = tl.load(
            value_cache_ptr + v_offs,
            mask=dim_mask[None, :],
            other=0.0
        )
        
        # Apply v_scale if needed
        if V.dtype.is_fp8():
            V = (V.to(tl.float32) * v_scale).to(Q.dtype)

        # Compute attention scores
        S = scale * tl.dot(Q, K)
        
        # Apply sequence masking
        seq_offs = block_idx * BLOCK_SIZE + token_offs
        seq_mask = seq_offs < seq_len
        S = tl.where(seq_mask[None, :], S, float("-inf"))
        
        # Apply sliding window if enabled
        if SLIDING_WINDOW > 0:
            context_pos = seq_len - 1
            is_in_window = (context_pos - seq_offs) < SLIDING_WINDOW
            S = tl.where(is_in_window[None, :], S, float("-inf"))

        # Apply alibi slopes if enabled
        if USE_ALIBI_SLOPES:
            pos_bias = seq_offs - (seq_len - 1)
            S += alibi_slope[:, None] * pos_bias[None, :]

        # Softmax reduction
        m_new = tl.maximum(M, tl.max(S, axis=1))
        alpha = tl.exp(M - m_new)
        p_curr = tl.exp(S - m_new[:, None])
        l_curr = tl.sum(p_curr, axis=1)
        
        # Update accumulator
        acc = acc * alpha[:, None]
        acc += tl.dot(p_curr.to(V.dtype), V)
        
        # Update running stats
        L = L * alpha + l_curr
        M = m_new

    # Final normalization
    acc = tl.where(L[:, None] > 0, acc / L[:, None], 0.0)

    # Store output
    out_offs = cur_batch_idx * output_stride_0 + query_head_idx[:, None] * output_stride_1
    tl.store(
        output_ptr + out_offs + dim_offs[None, :],
        acc,
        mask=head_mask[:, None] & dim_mask[None, :]
    )
"""

def main():
    io_manager = IOContractManager()
    cand_io = IOContract.from_dict({
        "args": [
            {"name": "output_ptr", "type": "tensor", "role": "output", "tensor_spec": {"shape": [132, 32, 64], "dtype": "float32"}}, 
            {"name": "query_ptr", "type": "tensor", "role": "input", "tensor_spec": {"shape": [132, 32, 64], "dtype": "float32", "init": {"kind": "randn", "seed": 42}}}, 
            {"name": "key_cache_ptr", "type": "tensor", "role": "input", "tensor_spec": {"shape": [8, 2, 64, 64, 1], "dtype": "float32", "init": {"kind": "randn", "seed": 43}}}, 
            {"name": "value_cache_ptr", "type": "tensor", "role": "input", "tensor_spec": {"shape": [8, 2, 64, 64], "dtype": "float32", "init": {"kind": "randn", "seed": 44}}}, 
            {"name": "sink_ptr", "type": "tensor", "role": "input", "tensor_spec": {"shape": [0], "dtype": "float32", "init": {"kind": "zeros"}}}, 
            {"name": "block_tables_ptr", "type": "tensor", "role": "input", "tensor_spec": {"shape": [132, 8], "dtype": "int32", "init": {"kind": "arange", "seed": 45}}}, 
            {"name": "seq_lens_ptr", "type": "tensor", "role": "input", "tensor_spec": {"shape": [132], "dtype": "int32", "init": {"kind": "full", "fill_value": 512}}}, 
            {"name": "alibi_slopes_ptr", "type": "tensor", "role": "input", "tensor_spec": {"shape": [0], "dtype": "float32", "init": {"kind": "zeros"}}}, 
            {"name": "scale", "type": "float", "value": 1.0, "role": "input"}, 
            {"name": "k_scale", "type": "float", "value": 1.0, "role": "input"}, 
            {"name": "v_scale", "type": "float", "value": 1.0, "role": "input"}, 
            {"name": "num_query_heads", "type": "int", "value": 32, "role": "input", "is_meta": True}, 
            {"name": "num_queries_per_kv", "type": "int", "value": 16, "role": "input", "is_meta": True}, 
            {"name": "num_queries_per_kv_padded", "type": "int", "value": 16, "role": "input", "is_meta": True}, 
            {"name": "block_table_stride", "type": "int", "value": 8, "role": "input"}, 
            {"name": "query_stride_0", "type": "int", "value": 2048, "role": "input"}, 
            {"name": "query_stride_1", "type": "int", "value": 64, "role": "input"}, 
            {"name": "output_stride_0", "type": "int", "value": 2048, "role": "input"}, 
            {"name": "output_stride_1", "type": "int", "value": 64, "role": "input"}, 
            {"name": "BLOCK_SIZE", "type": "int", "value": 64, "role": "input", "is_meta": True}, 
            {"name": "HEAD_SIZE", "type": "int", "value": 64, "role": "input", "is_meta": True}, 
            {"name": "HEAD_SIZE_PADDED", "type": "int", "value": 64, "role": "input", "is_meta": True}, 
            {"name": "USE_ALIBI_SLOPES", "type": "bool", "value": False, "role": "input", "is_meta": True}, 
            {"name": "SLIDING_WINDOW", "type": "int", "value": 0, "role": "input", "is_meta": True}, 
            {"name": "x", "type": "int", "value": 1, "role": "input", "is_meta": True}, 
            {"name": "stride_k_cache_0", "type": "int", "value": 8192, "role": "input"}, 
            {"name": "stride_k_cache_1", "type": "int", "value": 4096, "role": "input"}, 
            {"name": "stride_k_cache_2", "type": "int", "value": 64, "role": "input"}, 
            {"name": "stride_k_cache_3", "type": "int", "value": 1, "role": "input"}, 
            {"name": "stride_k_cache_4", "type": "int", "value": 1, "role": "input"}, 
            {"name": "stride_v_cache_0", "type": "int", "value": 8192, "role": "input"}, 
            {"name": "stride_v_cache_1", "type": "int", "value": 4096, "role": "input"}, 
            {"name": "stride_v_cache_2", "type": "int", "value": 64, "role": "input"}, 
            {"name": "stride_v_cache_3", "type": "int", "value": 1, "role": "input"}, 
            {"name": "filter_by_query_len", "type": "bool", "value": False, "role": "input", "is_meta": True}, 
            {"name": "query_start_len_ptr", "type": "tensor", "role": "input", "tensor_spec": {"shape": [0], "dtype": "int32", "init": {"kind": "zeros"}}}, 
            {"name": "USE_SINKS", "type": "bool", "value": False, "role": "input", "is_meta": True}
        ],
        "launch": {
            "grid": {"x": 132, "y": 2}, 
            "num_warps": 2, 
            "num_stages": 1
        }
    })

    ref_code = KernelCode(
        source_code=ref_src,
        kernel_type=KernelType.TRITON,
        io=cand_io
    )

    cand_code = KernelCode(
        source_code=cand_src,
        kernel_type=KernelType.TRITON,
        io=cand_io
    )
        
    cand_kernel = backend.compile(cand_code, gpu_id=0)
    ref_kernel = backend.compile(ref_code, gpu_id=0)
    validator = CorrectnessValidator()

    val = validator.validate_correctness(ref_kernel, cand_kernel, torch.device('cuda:0'), 2, 'abc')
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

main()    