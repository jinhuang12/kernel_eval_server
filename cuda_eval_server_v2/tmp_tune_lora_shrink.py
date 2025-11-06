# lora_shrink_autotune_bench.py
import math
import time
import torch
import triton
import triton.language as tl

# ----------------------------
# Config space (broad but sane)
# ----------------------------
AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK},
                  num_warps=w, num_stages=s)
    for BM in [32, 64, 96, 128]
    for BN in [16, 32, 64]
    for BK in [32, 64]
    for w  in [4, 8]
    for s  in [3, 4, 5]
]

# ----------------------------
# Heuristics and pruning
# ----------------------------

def _bytes_per_elem(dtype):
    # we assume fp16/bf16 for this kernel
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    return 4

def _smem_est_bytes(BM, BN, BK, stages, dtype_bytes):
    # rough staging footprint per block for tl.dot double-/multi-buffering
    return stages * (BM * BK + BK * BN) * dtype_bytes

def early_prune_one(config, **meta):
    """
    Called by Triton before compiling each candidate. Return True to PRUNE this config.
    meta contains runtime arguments (M,N,K, dtypes, etc.) and compile-time constants passed as heuristics.
    """
    BM = config.kwargs['BLOCK_M']
    BN = config.kwargs['BLOCK_N']
    BK = config.kwargs['BLOCK_K']
    w  = config.num_warps
    s  = config.num_stages

    M  = int(meta['M'])
    N  = int(meta['N'])
    K  = int(meta['K'])

    # 1) Require multiples of 16 for WGMMA-friendly tiling
    if (BM % 16) or (BN % 16) or (BK % 16):
        return True

    # 2) BN vs N: avoid launching wide-N tiles that mostly mask off columns
    if BN > max(64, 2*N):
        return True

    # 3) Avoid known high-reg patterns (seen in NCU): 8 warps + heavy BM*BK + BN>=32
    if (w == 8) and (BM * BK >= 8192) and (BN >= 32):
        return True

    # 4) Shared memory budget (rough). Keep generous headroom per block.
    dtype_bytes = 2  # fp16/bf16 assumed
    smem_est = _smem_est_bytes(BM, BN, BK, s, dtype_bytes)
    if smem_est > 98 * 1024:
        return True

    # 5) If N ≤ 32, we prefer BN close to N; allow BN=64 (2N) but discourage >64
    if N <= 32 and BN > 64:
        return True

    # 6) If M is very small, avoid overly huge BM (keeps waves/SM reasonable)
    if M < BM and BM >= 128:
        # if only one M tile would be launched, yield to smaller BM
        return True

    # We do not prune on EVEN_K; masking in K is allowed. Heuristics will set EVEN_K.
    return False


def early_config_prune(configs, named_args, **_):
    M = int(named_args['M']); N = int(named_args['N']); K = int(named_args['K'])
    keep = [c for c in configs if not early_prune_one(c, M=M, N=N, K=K)]
    return keep or [configs[0]]  # must return >=1 config

def perf_model(**kwargs):
    # Support both call styles: (config, named_args) or (config, **named_args)
    BM = kwargs['BLOCK_M']; BN =kwargs['BLOCK_N']; BK = kwargs['BLOCK_K']
    w  = int(kwargs['num_warps']);        s  = int(kwargs['num_stages'])
    K  = int(kwargs['K'])

    steps = (K + BK - 1) // BK
    bytes_per_kstep = (BM * BK + BK * BN) * 2  # fp16/bf16
    mem_cost = bytes_per_kstep * steps
    flops = 2.0 * BM * BN * K
    warp_penalty = 1.0 + 0.06 * (w // 4 - 1)
    stage_bonus  = 1.0 / (0.90 + 0.05 * s)
    return mem_cost * warp_penalty * stage_bonus + 0.05 * flops


PRUNE_CONFIGS_BY = {
    'early_config_prune': early_config_prune,
    'perf_model': perf_model,
    'top_k': 32,   # Only time the top-32 predicted configs
}

def _even_k_heur(args):
    bk = args.get('BLOCK_K')
    if bk is None:
        return False  # first pre-pass (before a specific config) — safe default
    return (int(args['K']) % int(bk)) == 0

# ----------------------------
# Kernels (with job-list scheduling)
# ----------------------------

@triton.jit
def mm_k(a_ptr, b_ptr, ak_stride, bk_stride, offset_k, M_LEN,
         K: tl.constexpr,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
         EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr, CAST_TYPE: tl.constexpr,
         b_dtype: tl.constexpr):
    """
    Inner K loop (accumulates into fp32). A/B are fp16/bf16; we mask rows by M_LEN to
    avoid wasted loads/compute for partial M tiles.
    """
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = tl.arange(0, BLOCK_M) < M_LEN
    for k in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            # A: random-gather → stream from L2; don't keep in L1
            tiled_a = tl.load(a_ptr, mask=mask_m[:, None], other=0,
                              cache_modifier=".cg")
            # B: reused within CTA across K; keep
            tiled_b = tl.load(b_ptr, cache_modifier=".ca")
        else:
            mask_a_k = mask_m[:, None] & (offset_k[None, :] < K - k * (BLOCK_K * SPLIT_K))
            tiled_a = tl.load(a_ptr, mask=mask_a_k, other=0,
                              cache_modifier=".cg")
            tiled_b = tl.load(b_ptr, mask=offset_k[:, None] < K - k * (BLOCK_K * SPLIT_K),
                              other=0, cache_modifier=".ca")
        if CAST_TYPE:
            tiled_a = tiled_a.to(b_dtype)
        accumulator += tl.dot(tiled_a, tiled_b)
        a_ptr += BLOCK_K * SPLIT_K * ak_stride
        b_ptr += BLOCK_K * SPLIT_K * bk_stride
    return accumulator


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
    # Select current lora pointer
    if SLICE_NUM == 1:
        cur_lora_ptr = lora_ptr
    else:
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(input_ptr.dtype.element_ty))

    # N tile
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    # K offsets and pointers
    offset_k = tl.arange(0, BLOCK_K) + pid_sk * BLOCK_K
    a_ptr = (input_ptr + ram[:, None] * input_d0_stride +
             offset_k[None, :] * input_d1_stride)
    b_ptr = (cur_lora_ptr + lora_d0_stride * lora_index +
             rbn[None, :] * lora_d1_stride +
             offset_k[:, None] * lora_d2_stride)

    # Compute partial/complete block
    acc = mm_k(a_ptr, b_ptr, input_d1_stride, lora_d2_stride, offset_k,
               M_LEN,
               K, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, SPLIT_K, False,
               cur_lora_ptr.dtype.element_ty)

    # Write-back
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_cm = tl.arange(0, BLOCK_M)
    cur_out_ptr = (out_ptr if SLICE_NUM == 1 else out_ptr + slice_id * output_d0_stride)
    c_ptr = cur_out_ptr + ram[:, None] * output_d1_stride + offset_cn[None, :] * output_d2_stride
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < N)
    acc *= scaling
    tl.store(c_ptr, acc, mask=c_mask)


# ---- Autotuned kernel wrapper (production path) ----

@triton.autotune(configs=AUTOTUNE_CONFIGS,
                 key=['M','N','K'],
                 prune_configs_by=PRUNE_CONFIGS_BY)
@triton.heuristics(values={
    'EVEN_K': _even_k_heur,
    'SPLIT_K': lambda args: 1,
})
@triton.jit
def _lora_shrink_kernel_auto(input_ptr, lora_ptr, out_ptr, M, N, K,
                             token_indices_sorted_by_lora_ids,
                             num_tokens_per_lora, lora_token_start_loc, lora_ids,
                             job_lora_idx, job_m_offsets, num_jobs,
                             scaling,
                             input_d0_stride, input_d1_stride,
                             lora_d0_stride, lora_d1_stride, lora_d2_stride,
                             output_d0_stride, output_d1_stride, output_d2_stride,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                             EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr, SLICE_NUM: tl.constexpr):

    cta_n_num = tl.cdiv(N, BLOCK_N)
    pid_sk_m_n = tl.program_id(axis=0)
    pid_sk = pid_sk_m_n % SPLIT_K
    pid_m_n = pid_sk_m_n // SPLIT_K
    pid_n = pid_m_n % cta_n_num
    pid_job = pid_m_n // cta_n_num

    if pid_sk * BLOCK_K >= K:
        return
    if pid_job >= num_jobs:
        return

    slice_id = tl.program_id(axis=1)

    # load job info
    lora_idx = tl.load(job_lora_idx + pid_job)
    cta_m_offset = tl.load(job_m_offsets + pid_job)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
    if cta_m_len <= 0:
        return

    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (token_indices_sorted_by_lora_ids +
                            lora_m_indices_start + cta_m_offset)

    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m).to(tl.int32)

    do_shrink_kernel(pid_n, pid_sk, slice_id, lora_id,
                     input_ptr, lora_ptr, out_ptr,
                     N, K, cta_m_len, ram,
                     input_d0_stride, input_d1_stride,
                     lora_d0_stride, lora_d1_stride, lora_d2_stride,
                     output_d0_stride, output_d1_stride, output_d2_stride,
                     scaling,
                     BLOCK_M, BLOCK_N, BLOCK_K,
                     EVEN_K, SPLIT_K, SLICE_NUM)


# ---- Non-autotuned variant for manual per-config benchmarking ----

@triton.jit
def _lora_shrink_kernel_cfg(input_ptr, lora_ptr, out_ptr, M, N, K,
                            token_indices_sorted_by_lora_ids,
                            num_tokens_per_lora, lora_token_start_loc, lora_ids,
                            job_lora_idx, job_m_offsets, num_jobs,
                            scaling,
                            input_d0_stride, input_d1_stride,
                            lora_d0_stride, lora_d1_stride, lora_d2_stride,
                            output_d0_stride, output_d1_stride, output_d2_stride,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                            EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr, SLICE_NUM: tl.constexpr):

    cta_n_num = tl.cdiv(N, BLOCK_N)
    pid_sk_m_n = tl.program_id(axis=0)
    pid_sk = pid_sk_m_n % SPLIT_K
    pid_m_n = pid_sk_m_n // SPLIT_K
    pid_n = pid_m_n % cta_n_num
    pid_job = pid_m_n // cta_n_num

    if pid_sk * BLOCK_K >= K:
        return
    if pid_job >= num_jobs:
        return

    slice_id = tl.program_id(axis=1)

    lora_idx = tl.load(job_lora_idx + pid_job)
    cta_m_offset = tl.load(job_m_offsets + pid_job)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
    if cta_m_len <= 0:
        return

    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (token_indices_sorted_by_lora_ids +
                            lora_m_indices_start + cta_m_offset)

    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m).to(tl.int32)

    do_shrink_kernel(pid_n, pid_sk, slice_id, lora_id,
                     input_ptr, lora_ptr, out_ptr,
                     N, K, cta_m_len, ram,
                     input_d0_stride, input_d1_stride,
                     lora_d0_stride, lora_d1_stride, lora_d2_stride,
                     output_d0_stride, output_d1_stride, output_d2_stride,
                     scaling,
                     BLOCK_M, BLOCK_N, BLOCK_K,
                     (K % BLOCK_K) == 0, 1, SLICE_NUM=1)

# ----------------------------
# Host helpers: build LoRA metadata, compact, job list
# ----------------------------

def _get_lora_a_ptr(lora_a_weights, device):
    # same semantics as in your snippet
    lora_strides_d0 = []
    lora_strides_d1 = []
    lora_strides_d2 = []
    tensor_ptrs = []
    for lora_a_weight in lora_a_weights:
        t = lora_a_weight
        if t.ndim == 4:  # (lora_num,1,size,rank)
            assert t.size(1) == 1
            t = t.squeeze(1)
        else:
            assert t.ndim == 3
        assert t.is_contiguous()
        tensor_ptrs.append(t.data_ptr())
        lora_strides_d0.append(t.stride(0))
        lora_strides_d1.append(t.stride(1))
        lora_strides_d2.append(t.stride(2))
    if len(lora_a_weights) > 1:
        lora_ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)
    else:
        lora_ptr_tensor = lora_a_weights[0]
    assert len(set(lora_strides_d0)) == 1 and len(set(lora_strides_d1)) == 1 and len(set(lora_strides_d2)) == 1
    return lora_ptr_tensor, lora_strides_d0[0], lora_strides_d1[0], lora_strides_d2[0]

def compact_and_sort_per_lora(token_indices_sorted_by_lora_ids,
                              padded_num_tokens_per_lora,
                              padded_lora_ids):
    # compact away -1 and zero-count
    active_mask = (padded_lora_ids != -1) & (padded_num_tokens_per_lora > 0)
    lora_ids_compact = padded_lora_ids[active_mask]
    num_tokens_per_lora_compact = padded_num_tokens_per_lora[active_mask]

    # build start locations
    if num_tokens_per_lora_compact.numel() > 0:
        lora_token_start_loc_compact = torch.zeros_like(num_tokens_per_lora_compact)
        lora_token_start_loc_compact[1:] = torch.cumsum(num_tokens_per_lora_compact[:-1], 0)
    else:
        lora_token_start_loc_compact = torch.zeros(0, dtype=torch.int32, device=padded_lora_ids.device)

    # sort tokens within each active lora by row id to improve locality
    tis = token_indices_sorted_by_lora_ids.clone()
    for i in range(num_tokens_per_lora_compact.numel()):
        start = int(lora_token_start_loc_compact[i].item())
        count = int(num_tokens_per_lora_compact[i].item())
        if count > 1:
            seg = tis[start:start+count]
            order = torch.argsort(seg)
            tis[start:start+count] = seg[order]
    return tis, num_tokens_per_lora_compact, lora_token_start_loc_compact, lora_ids_compact

def build_job_list(num_tokens_per_lora_compact, BLOCK_M, device):
    # make one job per real M block for each active lora
    if num_tokens_per_lora_compact.numel() == 0:
        return torch.empty(0, dtype=torch.int32, device=device), torch.empty(0, dtype=torch.int32, device=device)
    blocks_per_lora = (num_tokens_per_lora_compact + BLOCK_M - 1) // BLOCK_M
    num_jobs = int(blocks_per_lora.sum().item())
    job_lora_idx = torch.empty(num_jobs, dtype=torch.int32, device=device)
    job_m_offsets = torch.empty(num_jobs, dtype=torch.int32, device=device)
    write = 0
    for i in range(blocks_per_lora.numel()):
        nblk = int(blocks_per_lora[i].item())
        if nblk == 0:
            continue
        job_lora_idx[write:write+nblk] = i
        job_m_offsets[write:write+nblk] = torch.arange(nblk, device=device, dtype=torch.int32) * BLOCK_M
        write += nblk
    return job_lora_idx, job_m_offsets

# ----------------------------
# Manual benchmark over pruned configs
# ----------------------------

def generate_pruned_cfgs(M, N, K):
    pruned = []
    for cfg in AUTOTUNE_CONFIGS:
        if not early_prune_one(cfg, M=M, N=N, K=K):
            pruned.append(cfg)
    # Sort using the SAME calling convention autotune uses
    pruned.sort(key=lambda cfg: perf_model(M=M, N=N, K=K, **cfg.all_kwargs()))
    return pruned[:PRUNE_CONFIGS_BY['top_k']]


def launch_and_time(cfg, tensors, warmup=5, iters=50):
    device = tensors['inputs'].device
    torch.cuda.synchronize(device)

    inputs = tensors['inputs']
    lora_ptr_tensor = tensors['lora_ptr_tensor']
    output_tensor = tensors['output_tensor']
    token_indices_sorted_by_lora_ids = tensors['token_indices_sorted_by_lora_ids']
    num_tokens_per_lora = tensors['num_tokens_per_lora_compact']
    lora_token_start_loc = tensors['lora_token_start_loc_compact']
    lora_ids = tensors['lora_ids_compact']
    scaling = tensors['scaling']

    M, K = inputs.shape
    num_slices = 1
    N = output_tensor.size(-1)

    # BLOCK params
    BM = cfg.kwargs['BLOCK_M']
    BN = cfg.kwargs['BLOCK_N']
    BK = cfg.kwargs['BLOCK_K']
    EVEN_K = (K % BK) == 0

    # job list depends on BM
    job_lora_idx, job_m_offsets = build_job_list(num_tokens_per_lora, BM, device)
    num_jobs = job_lora_idx.numel()

    grid_x = max(1, num_jobs) * ((N + BN - 1) // BN)  # SPLIT_K=1
    grid = (grid_x, num_slices, 1)

    # Warmup
    for _ in range(warmup):
        _lora_shrink_kernel_cfg[grid](
            inputs, lora_ptr_tensor, output_tensor,
            M, N, K,
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora, lora_token_start_loc, lora_ids,
            job_lora_idx, job_m_offsets, num_jobs,
            scaling,
            inputs.stride(0), inputs.stride(1),
            tensors['lora_strides_d0'], tensors['lora_strides_d1'], tensors['lora_strides_d2'],
            output_tensor.stride(0), output_tensor.stride(1), output_tensor.stride(2),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            EVEN_K=EVEN_K, SPLIT_K=1, SLICE_NUM=1,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
        )
    torch.cuda.synchronize(device)

    # Timed
    times = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_evt.record()
        _lora_shrink_kernel_cfg[grid](
            inputs, lora_ptr_tensor, output_tensor,
            M, N, K,
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora, lora_token_start_loc, lora_ids,
            job_lora_idx, job_m_offsets, num_jobs,
            scaling,
            inputs.stride(0), inputs.stride(1),
            tensors['lora_strides_d0'], tensors['lora_strides_d1'], tensors['lora_strides_d2'],
            output_tensor.stride(0), output_tensor.stride(1), output_tensor.stride(2),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            EVEN_K=EVEN_K, SPLIT_K=1, SLICE_NUM=1,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
        )
        end_evt.record()
        end_evt.synchronize()
        times.append(start_evt.elapsed_time(end_evt))  # ms

    # p50
    times_sorted = sorted(times)
    p50 = times_sorted[len(times_sorted) // 2]
    return p50, min(times_sorted)

# ----------------------------
# Test data creation (similar to your snippet)
# ----------------------------

def make_test(num_tokens=4096, hidden_size=2048, lora_rank=32,
              num_loras=16, num_slices=1, dtype=torch.float16, device=None):
    torch.manual_seed(0)
    if device is None:
        device = torch.device('cuda')

    # Input
    inputs = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # LoRA A weights: [num_slices][num_loras, lora_rank, hidden_size]
    lora_a_weights = [torch.randn(num_loras, lora_rank, hidden_size,
                                  dtype=dtype, device=device) for _ in range(num_slices)]

    # Output: [num_slices, num_tokens, lora_rank]
    output_tensor = torch.zeros(num_slices, num_tokens, lora_rank, dtype=dtype, device=device)

    # token -> LoRA mapping, allow some -1 (no LoRA)
    token_lora_mapping = torch.randint(-1, num_loras, (num_tokens,), device=device)

    # build metadata like your snippet
    _, token_indices_sorted_by_lora_ids = torch.sort(token_lora_mapping, stable=True)
    active_lora_ids, num_tokens_per_lora = torch.unique(token_lora_mapping, sorted=True, return_counts=True)

    max_loras = num_loras
    padded_num_tokens_per_lora = torch.zeros(max_loras, dtype=torch.int32, device=device)
    actual_num_loras = min(len(num_tokens_per_lora), max_loras)
    if actual_num_loras > 0:
        padded_num_tokens_per_lora[:actual_num_loras] = num_tokens_per_lora[:actual_num_loras]

    padded_lora_ids = torch.full((max_loras,), -1, dtype=torch.int32, device=device)
    if actual_num_loras > 0:
        padded_lora_ids[:actual_num_loras] = active_lora_ids[:actual_num_loras]

    # Compact + sort within loRA
    tis, num_tokens_per_lora_compact, lora_token_start_loc_compact, lora_ids_compact = \
        compact_and_sort_per_lora(token_indices_sorted_by_lora_ids,
                                  padded_num_tokens_per_lora,
                                  padded_lora_ids)

    # pointers/strides
    lora_ptr_tensor, l0, l1, l2 = _get_lora_a_ptr(lora_a_weights, inputs.device)

    return {
        'inputs': inputs,
        'lora_a_weights': lora_a_weights,
        'output_tensor': output_tensor,
        'token_lora_mapping': token_lora_mapping,
        'token_indices_sorted_by_lora_ids': tis,
        'num_tokens_per_lora_compact': num_tokens_per_lora_compact,
        'lora_token_start_loc_compact': lora_token_start_loc_compact,
        'lora_ids_compact': lora_ids_compact,
        'lora_ptr_tensor': lora_ptr_tensor,
        'lora_strides_d0': l0, 'lora_strides_d1': l1, 'lora_strides_d2': l2,
        'scaling': 1.0,
    }

# ----------------------------
# Main: run manual bench and report best; also demo autotuned call
# ----------------------------

def main():
    assert torch.cuda.is_available(), "CUDA device required"
    device = torch.device('cuda')
    tensors = make_test(device=device)
    inputs = tensors['inputs']
    output = tensors['output_tensor']
    M, K = inputs.shape
    N = output.size(-1)

    print(f"Problem: M={M}, K={K}, N={N}  (LoRA rank)")
    pruned_cfgs = generate_pruned_cfgs(M, N, K)
    print(f"Total candidate configs: {len(AUTOTUNE_CONFIGS)}")
    print(f"Pruned to top-{len(pruned_cfgs)} by heuristics/model: {len(pruned_cfgs)}")

    rows = []
    for cfg in pruned_cfgs:
        p50, best = launch_and_time(cfg, tensors)
        rows.append((p50, best, cfg))
        print(f"CFG  BM={cfg.kwargs['BLOCK_M']:>3} BN={cfg.kwargs['BLOCK_N']:>3} BK={cfg.kwargs['BLOCK_K']:>3} "
              f"W={cfg.num_warps} S={cfg.num_stages}  ->  p50={p50:.3f} ms (best={best:.3f} ms)")

    rows.sort(key=lambda x: x[0])
    best_p50, best_best, best_cfg = rows[0]
    print("\n==== Best config (manual bench) ====")
    print(f"BM={best_cfg.kwargs['BLOCK_M']} BN={best_cfg.kwargs['BLOCK_N']} BK={best_cfg.kwargs['BLOCK_K']} "
          f"W={best_cfg.num_warps} S={best_cfg.num_stages}  ->  p50={best_p50:.3f} ms (best={best_best:.3f} ms)")

    # Optional: demonstrate a single call through the AUTOTUNED kernel (production)
    # Build job list for the best BM (the autotuner will choose its own)
    BM = best_cfg.kwargs['BLOCK_M']
    job_lora_idx, job_m_offsets = build_job_list(tensors['num_tokens_per_lora_compact'], BM, device)
    num_jobs = job_lora_idx.numel()
    # BN-agnostic grid: ensure enough CTAs regardless of which BN the autotuner picks.
    MIN_BLOCK_N = min(cfg.kwargs['BLOCK_N'] for cfg in AUTOTUNE_CONFIGS)  # -> 16 in this search space
    grid_x = max(1, num_jobs) * ((N + MIN_BLOCK_N - 1) // MIN_BLOCK_N)
    grid = (grid_x, 1, 1)
    
    torch.cuda.synchronize(device)
    _lora_shrink_kernel_auto[grid](
        tensors['inputs'], tensors['lora_ptr_tensor'], tensors['output_tensor'],
        M, N, K,
        tensors['token_indices_sorted_by_lora_ids'],
        tensors['num_tokens_per_lora_compact'],
        tensors['lora_token_start_loc_compact'],
        tensors['lora_ids_compact'],
        job_lora_idx, job_m_offsets, num_jobs,
        tensors['scaling'],
        tensors['inputs'].stride(0), tensors['inputs'].stride(1),
        tensors['lora_strides_d0'], tensors['lora_strides_d1'], tensors['lora_strides_d2'],
        tensors['output_tensor'].stride(0), tensors['output_tensor'].stride(1), tensors['output_tensor'].stride(2),
        SLICE_NUM=1
    )
    torch.cuda.synchronize(device)
    print("\n(Autotuned path executed once to cache best config; see manual results above for full timing.)")

if __name__ == "__main__":
    main()
