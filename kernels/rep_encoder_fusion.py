import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_stats_kernel(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xh: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    mask = offs_h < H
    x = tl.load(x_ptr + row * stride_xm + offs_h * stride_xh, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)
    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    mean = sum_x / H
    var = sum_x2 / H - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = tl.rsqrt(var + EPS)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _layernorm_linear_kernel(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    gamma_ptr,
    beta_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BF16_DOT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mean = tl.load(mean_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xh
    w_ptrs = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wh
    for _ in range(0, H, BLOCK_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < H),
            other=0.0,
        ).to(tl.float32)
        gamma = tl.load(gamma_ptr + offs_k, mask=offs_k < H, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + offs_k, mask=offs_k < H, other=0.0).to(tl.float32)
        x = (x - mean[:, None]) * rstd[:, None]
        x = x * gamma[None, :] + beta[None, :]
        w = tl.load(
            w_ptrs,
            mask=(offs_k[:, None] < H) & (offs_n[None, :] < N),
            other=0.0,
        )
        if BF16_DOT:
            x = x.to(tl.bfloat16)
            w = w.to(tl.bfloat16)
        else:
            w = w.to(tl.float32)
        acc += tl.dot(x, w, out_dtype=tl.float32)
        x_ptrs += BLOCK_K * stride_xh
        w_ptrs += BLOCK_K * stride_wh
        offs_k += BLOCK_K

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def cat_free_layernorm_linear(pooled_embs, layernorm, linear):
    """Reference implementation for cat + LayerNorm + Linear without cat.

    This is an accuracy-first prototype. It avoids materializing the wide
    concatenated activation, but still uses multiple PyTorch matmuls, so it is
    mainly a stepping stone for a later Triton fused kernel.
    """
    if not pooled_embs:
        raise ValueError("pooled_embs must be non-empty")

    rows = pooled_embs[0].shape[0]
    device = pooled_embs[0].device
    total_dim = sum(x.shape[1] for x in pooled_embs)
    acc_dtype = torch.float32

    sum_x = torch.zeros(rows, device=device, dtype=acc_dtype)
    sum_x2 = torch.zeros(rows, device=device, dtype=acc_dtype)
    for x in pooled_embs:
        x_float = x.float()
        sum_x += x_float.sum(dim=1)
        sum_x2 += (x_float * x_float).sum(dim=1)

    mean = sum_x / total_dim
    var = torch.clamp(sum_x2 / total_dim - mean * mean, min=0.0)
    rstd = torch.rsqrt(var + layernorm.eps)

    if linear.bias is None:
        out = torch.zeros(rows, linear.out_features, device=device, dtype=acc_dtype)
    else:
        out = linear.bias.float().expand(rows, -1).clone()

    ln_weight = layernorm.weight.float() if layernorm.weight is not None else None
    ln_bias = layernorm.bias.float() if layernorm.bias is not None else None
    linear_weight = linear.weight.float()

    col_start = 0
    for x in pooled_embs:
        width = x.shape[1]
        col_end = col_start + width
        x_norm = (x.float() - mean[:, None]) * rstd[:, None]
        if ln_weight is not None:
            x_norm = x_norm * ln_weight[col_start:col_end]
        if ln_bias is not None:
            x_norm = x_norm + ln_bias[col_start:col_end]
        out += x_norm @ linear_weight[:, col_start:col_end].t()
        col_start = col_end

    return out.to(dtype=linear.weight.dtype)


def _next_power_of_2(x):
    return 1 << (int(x) - 1).bit_length()


def layernorm_linear_triton(
    x,
    layernorm,
    linear,
    block_m=16,
    block_n=16,
    block_k=64,
    group_m=8,
):
    """Triton fused LayerNorm + Linear for a contiguous [M, H] input."""
    if not x.is_cuda:
        raise RuntimeError("layernorm_linear_triton expects a CUDA tensor")
    if x.dim() != 2:
        raise ValueError("layernorm_linear_triton expects x as [M, H]")
    if layernorm.weight is None or layernorm.bias is None:
        raise ValueError("layernorm_linear_triton requires affine LayerNorm")

    x = x.contiguous()
    weight = linear.weight.contiguous()
    bias = None if linear.bias is None else linear.bias.contiguous()
    gamma = layernorm.weight.contiguous()
    beta = layernorm.bias.contiguous()

    m, h = x.shape
    n, wh = weight.shape
    if wh != h:
        raise ValueError(f"shape mismatch: x H={h}, weight H={wh}")

    mean = torch.empty((m,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((m,), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    block_h = _next_power_of_2(h)
    _layernorm_stats_kernel[(m,)](
        x,
        mean,
        rstd,
        m,
        h,
        x.stride(0),
        x.stride(1),
        EPS=layernorm.eps,
        BLOCK_H=block_h,
        num_warps=8,
    )

    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _layernorm_linear_kernel[grid](
        x,
        mean,
        rstd,
        gamma,
        beta,
        weight,
        bias if bias is not None else out,
        out,
        m,
        h,
        n,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        BF16_DOT=x.dtype is torch.bfloat16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=4,
        num_stages=4,
    )
    return out
