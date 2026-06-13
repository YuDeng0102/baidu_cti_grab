import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _quantize_rows_int8_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < COLS

    x = tl.load(x_ptr + row * stride_xm + offs_k * stride_xk, mask=mask, other=0.0).to(tl.float32)
    abs_x = tl.abs(x)
    max_abs = tl.max(tl.where(mask, abs_x, 0.0), axis=0)
    scale = tl.maximum(max_abs / 127.0, 1.0e-8)

    q = x / scale
    q = tl.where(q >= 0.0, q + 0.5, q - 0.5)
    q = q.to(tl.int32)
    q = tl.minimum(tl.maximum(q, -127), 127)

    tl.store(q_ptr + row * stride_qm + offs_k * stride_qk, q.to(tl.int8), mask=mask)
    tl.store(scale_ptr + row, scale)


@triton.jit
def _w8a8_matmul_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    a_scale_ptr,
    w_scale_ptr,
    bias_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
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

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0,
        )
        w = tl.load(
            w_ptrs,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )
        acc += tl.dot(a, w, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
        offs_k += BLOCK_K

    a_scale = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    w_scale = tl.load(w_scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    c = acc.to(tl.float32) * a_scale[:, None] * w_scale[None, :]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        c += bias[None, :]

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _w8a8_dynamic_fused_kernel(
    x_ptr,
    w_ptr,
    c_ptr,
    w_scale_ptr,
    bias_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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

    # First pass: calculate one dynamic scale per input row.
    row_max = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(tl.abs(x), axis=1))
    x_scale = tl.maximum(row_max / 127.0, 1.0e-8)

    # Second pass: quantize each activation tile in registers and immediately
    # consume it in INT8 Tensor Core matmul. No global x_int8 tensor is written.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        q = x / x_scale[:, None]
        q = tl.where(q >= 0.0, q + 0.5, q - 0.5)
        q = q.to(tl.int32)
        q = tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)

        w = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + k_idxs[:, None] * stride_wk,
            mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
            other=0,
        )
        acc += tl.dot(q, w, out_dtype=tl.int32)

    w_scale = tl.load(w_scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    c = acc.to(tl.float32) * x_scale[:, None] * w_scale[None, :]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        c += bias[None, :]

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


_W8A8_MATMUL_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 4},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
]


_w8a8_matmul_kernel_autotuned = None


def _get_w8a8_matmul_kernel_autotuned():
    global _w8a8_matmul_kernel_autotuned
    if _w8a8_matmul_kernel_autotuned is None:
        _w8a8_matmul_kernel_autotuned = triton.autotune(
            configs=_W8A8_MATMUL_AUTOTUNE_CONFIGS,
            key=["M", "N", "K", "HAS_BIAS"],
        )(_w8a8_matmul_kernel)
    return _w8a8_matmul_kernel_autotuned



def _next_power_of_2(x):
    return 1 << (int(x) - 1).bit_length()


def quantize_rows_int8_triton(x):
    if not x.is_cuda:
        raise RuntimeError("quantize_rows_int8_triton expects a CUDA tensor")
    if x.dim() != 2:
        raise ValueError("quantize_rows_int8_triton expects [rows, cols]")

    x = x.contiguous()
    rows, cols = x.shape
    q = torch.empty((rows, cols), device=x.device, dtype=torch.int8)
    scale = torch.empty((rows,), device=x.device, dtype=torch.float32)
    block_k = _next_power_of_2(cols)

    _quantize_rows_int8_kernel[(rows,)](
        x,
        q,
        scale,
        rows,
        cols,
        x.stride(0),
        x.stride(1),
        q.stride(0),
        q.stride(1),
        BLOCK_K=block_k,
        num_warps=8,
    )
    return q, scale


def quantize_weight_per_channel_int8(weight):
    """Triton symmetric per-output-channel INT8 quantization for Linear weights."""
    return quantize_rows_int8_triton(weight)


def quantize_activation_per_token_int8(x):
    """Triton symmetric per-token INT8 activation quantization."""
    return quantize_rows_int8_triton(x)


def w8a8_linear_triton_dynamic_unfused(
    x,
    weight_int8,
    weight_scale,
    bias=None,
    out_dtype=torch.bfloat16,
    block_m=32,
    block_n=64,
    block_k=64,
    group_m=8,
    autotune=False,
):
    """Dynamic W8A8 Linear using separate activation quantization and GEMM kernels.

    Args:
        x: [M, K] floating point activation.
        weight_int8: [N, K] int8 weight.
        weight_scale: [N] fp32 per-output-channel scale.
        bias: optional [N] bias.
    """
    if not x.is_cuda:
        raise RuntimeError("w8a8_linear_triton_dynamic_unfused expects CUDA tensors")
    if x.dim() != 2 or weight_int8.dim() != 2:
        raise ValueError("expected x [M,K] and weight_int8 [N,K]")

    x = x.contiguous()
    weight_int8 = weight_int8.contiguous()
    weight_scale = weight_scale.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    m, k = x.shape
    n, wk = weight_int8.shape
    if wk != k:
        raise ValueError(f"shape mismatch: x K={k}, weight K={wk}")

    x_int8, x_scale = quantize_activation_per_token_int8(x)
    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    kernel = _get_w8a8_matmul_kernel_autotuned() if autotune else _w8a8_matmul_kernel
    launch_grid = (lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)) if autotune else grid
    kernel[launch_grid](
        x_int8,
        weight_int8,
        out,
        x_scale,
        weight_scale,
        bias if bias is not None else out,
        m,
        n,
        k,
        x_int8.stride(0),
        x_int8.stride(1),
        weight_int8.stride(0),
        weight_int8.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        **({} if autotune else {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "num_warps": 4,
            "num_stages": 4,
        }),
    )
    return out


def w8a8_linear_triton_dynamic_fused(
    x,
    weight_int8,
    weight_scale,
    bias=None,
    out_dtype=torch.bfloat16,
    block_m=32,
    block_n=64,
    block_k=64,
    group_m=8,
):
    """Single-kernel dynamic W8A8 Linear with on-the-fly activation quantization."""
    if not x.is_cuda:
        raise RuntimeError("w8a8_linear_triton_dynamic_fused expects CUDA tensors")
    if x.dim() != 2 or weight_int8.dim() != 2:
        raise ValueError("expected x [M,K] and weight_int8 [N,K]")

    x = x.contiguous()
    weight_int8 = weight_int8.contiguous()
    weight_scale = weight_scale.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    m, k = x.shape
    n, wk = weight_int8.shape
    if wk != k:
        raise ValueError(f"shape mismatch: x K={k}, weight K={wk}")

    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _w8a8_dynamic_fused_kernel[grid](
        x,
        weight_int8,
        out,
        weight_scale,
        bias if bias is not None else out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        weight_int8.stride(0),
        weight_int8.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=4,
        num_stages=4,
    )
    return out


def w8a8_linear_dynamic(*args, **kwargs):
    return w8a8_linear_triton_dynamic_fused(*args, **kwargs)


def bf16_linear(x, weight, bias=None):
    return F.linear(
        x.to(torch.bfloat16),
        weight.to(torch.bfloat16),
        None if bias is None else bias.to(torch.bfloat16),
    )


def max_abs_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def mean_abs_diff(a, b):
    return (a.float() - b.float()).abs().mean().item()
