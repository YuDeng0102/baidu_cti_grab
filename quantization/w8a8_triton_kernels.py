# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for W8A8 INT8 Linear."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    import w8a8_cuda_quant as _cuda_quant
    _cuda_quant_import_error = None
except Exception as exc:
    _cuda_quant = None
    _cuda_quant_import_error = exc

from w8a8_preprocess import QuantizedWeight





@triton.jit
def _static_quant_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < total
    inv_scale = 1.0 / tl.load(scale_ptr)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    xq = tl.extra.cuda.libdevice.round(x * inv_scale)
    xq = tl.minimum(tl.maximum(xq, -128.0), 127.0).to(tl.int8)
    tl.store(xq_ptr + offs, xq, mask=mask)


@triton.jit
def _dynamic_per_token_quant_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    mask = offs_k < K
    x = tl.load(x_ptr + row * stride_xm + offs_k * stride_xk, mask=mask, other=0.0)
    x = x.to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-8)
    scale = amax / 127.0
    inv_scale = 127.0 / amax
    xq = tl.extra.cuda.libdevice.round(x * inv_scale)
    xq = tl.minimum(tl.maximum(xq, -128.0), 127.0).to(tl.int8)
    tl.store(xq_ptr + row * stride_qm + offs_k * stride_qk, xq, mask=mask)
    tl.store(scale_ptr + row, scale)


@triton.jit(do_not_specialize=["M"])
def _int8_mm_no_scale_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
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
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    masks_m = offs_m < M
    masks_n = offs_n < N

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        masks_k = offs_k < K
        a = tl.load(a_ptrs, mask=masks_m[:, None] & masks_k[None, :], other=0)
        b = tl.load(b_ptrs, mask=masks_k[:, None] & masks_n[None, :], other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        offs_k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=masks_m[:, None] & masks_n[None, :])


@triton.jit(do_not_specialize=["M"])
def _int8_mm_no_scale_nomask_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
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
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    masks_m = offs_m < M

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=masks_m[:, None], other=0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=masks_m[:, None])


@triton.jit(do_not_specialize=["M"])
def _w8a8_scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    STATIC_ACT: tl.constexpr,
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
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    masks_m = offs_m < M
    masks_n = offs_n < N

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        masks_k = offs_k < K
        a = tl.load(a_ptrs, mask=masks_m[:, None] & masks_k[None, :], other=0)
        b = tl.load(b_ptrs, mask=masks_k[:, None] & masks_n[None, :], other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        offs_k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc_f = acc.to(tl.float32)
    if STATIC_ACT:
        scale_a = tl.load(scale_a_ptr)
        acc_f = acc_f * scale_a
    else:
        scale_a = tl.load(scale_a_ptr + offs_m, mask=masks_m, other=0.0)
        acc_f = acc_f * scale_a[:, None]

    scale_b = tl.load(scale_b_ptr + offs_n, mask=masks_n, other=0.0)
    out = acc_f * scale_b[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=masks_n, other=0.0).to(tl.float32)
        out += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=masks_m[:, None] & masks_n[None, :])


@triton.jit(do_not_specialize=["M"])
def _w8a8_scaled_mm_nomask_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    STATIC_ACT: tl.constexpr,
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
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    masks_m = offs_m < M

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=masks_m[:, None], other=0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc_f = acc.to(tl.float32)
    if STATIC_ACT:
        scale_a = tl.load(scale_a_ptr)
        acc_f = acc_f * scale_a
    else:
        scale_a = tl.load(scale_a_ptr + offs_m, mask=masks_m, other=0.0)
        acc_f = acc_f * scale_a[:, None]

    scale_b = tl.load(scale_b_ptr + offs_n)
    out = acc_f * scale_b[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n).to(tl.float32)
        out += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=masks_m[:, None])


def flatten_input(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Flatten [..., K] -> [M, K] and keep the original leading shape."""
    return x.contiguous().view(-1, x.shape[-1]), x.shape[:-1]


def _cuda_quant_ops_available() -> bool:
    return _cuda_quant is not None and _cuda_quant.is_available()


def quant_backend_name() -> str:
    return "cuda_ext" if _cuda_quant_ops_available() else "triton"


def quant_backend_error() -> str | None:
    if _cuda_quant is None:
        return None if _cuda_quant_import_error is None else repr(_cuda_quant_import_error)
    if not hasattr(_cuda_quant, "load_error_message"):
        return None
    return _cuda_quant.load_error_message()


def _cuda_static_quantize_activation_out(
    x_2d: torch.Tensor,
    scale: torch.Tensor,
    xq: torch.Tensor,
) -> torch.Tensor:
    assert _cuda_quant is not None
    return _cuda_quant.static_quantize_activation_out(x_2d, scale, xq)


def _cuda_static_quantize_activation(
    x_2d: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    assert _cuda_quant is not None
    return _cuda_quant.static_quantize_activation(x_2d, scale)


def _cuda_dynamic_per_token_quantize_activation(
    x_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert _cuda_quant is not None
    return _cuda_quant.dynamic_per_token_quantize_activation(x_2d)


def _cuda_dynamic_per_token_quantize_activation_out(
    x_2d: torch.Tensor,
    xq: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert _cuda_quant is not None
    return _cuda_quant.dynamic_per_token_quantize_activation_out(x_2d, xq, scale)


def static_quantize_activation_out(
    x_2d: torch.Tensor,
    scale: torch.Tensor,
    xq: torch.Tensor,
) -> torch.Tensor:
    total = x_2d.numel()
    block = 1024
    grid = (triton.cdiv(total, block),)
    _static_quant_kernel[grid](x_2d, xq, scale, total, BLOCK=block)
    return xq


def static_quantize_activation(x_2d: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    xq = torch.empty_like(x_2d, dtype=torch.int8)
    return static_quantize_activation_out(x_2d, scale, xq)


def dynamic_per_token_quantize_activation_out(
    x_2d: torch.Tensor,
    xq: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, K = x_2d.shape
    block_k = triton.next_power_of_2(K)
    if block_k > 65536:
        raise ValueError(f"K={K} is too large for this simple one-program-per-row kernel")
    _dynamic_per_token_quant_kernel[(M,)](
        x_2d,
        xq,
        scale,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        xq.stride(0),
        xq.stride(1),
        BLOCK_K=block_k,
        num_warps=min(max(block_k // 256, 1), 8),
    )
    return xq, scale


def dynamic_per_token_quantize_activation(
    x_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = x_2d.shape
    xq = torch.empty_like(x_2d, dtype=torch.int8)
    scale = torch.empty((M,), device=x_2d.device, dtype=torch.float32)
    return dynamic_per_token_quantize_activation_out(x_2d, xq, scale)


def quantize_activation_out(
    x_2d: torch.Tensor,
    xq: torch.Tensor,
    scale_out: torch.Tensor,
    *,
    act_mode: str,
    static_act_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert act_mode in {"static", "dynamic"}
    use_cuda_ext = _cuda_quant_ops_available()
    if act_mode == "static":
        assert static_act_scale is not None
        if use_cuda_ext:
            return _cuda_static_quantize_activation_out(x_2d, static_act_scale, xq), static_act_scale
        return static_quantize_activation_out(x_2d, static_act_scale, xq), static_act_scale
    if use_cuda_ext:
        return _cuda_dynamic_per_token_quantize_activation_out(x_2d, xq, scale_out)
    return dynamic_per_token_quantize_activation_out(x_2d, xq, scale_out)


def quantize_activation(
    x_2d: torch.Tensor,
    *,
    act_mode: str,
    static_act_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert act_mode in {"static", "dynamic"}
    use_cuda_ext = _cuda_quant_ops_available()
    if act_mode == "static":
        assert static_act_scale is not None
        if use_cuda_ext:
            return _cuda_static_quantize_activation(x_2d, static_act_scale), static_act_scale
        return static_quantize_activation(x_2d, static_act_scale), static_act_scale
    if use_cuda_ext:
        return _cuda_dynamic_per_token_quantize_activation(x_2d)
    return dynamic_per_token_quantize_activation(x_2d)



def _can_use_nomask_kernel(K: int, N: int) -> bool:
    # The nomask autotune pool contains BLOCK_K/BLOCK_N up to 256.  Requiring
    # divisibility by 256 keeps every candidate config safe from K/N OOB loads.
    return K % 256 == 0 and N % 256 == 0


def w8a8_scaled_mm_out(
    xq_2d: torch.Tensor,
    qweight: QuantizedWeight,
    scale_a: torch.Tensor,
    out_2d: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    static_act: bool,
) -> torch.Tensor:
    assert xq_2d.ndim == 2
    K = xq_2d.shape[1]
    assert K == qweight.qweight_t.shape[0]

    M = xq_2d.shape[0]
    N = qweight.qweight_t.shape[1]
    
    # --- Shape-based Heuristic Dispatcher ---
    block_n = min(128, triton.next_power_of_2(N))
    block_k = min(128, triton.next_power_of_2(K))
    if M < 2048:
        block_m = 32
        num_warps = 4
        num_stages = 3
    elif M < 4096:
        block_m = 128
        num_warps = 8
        num_stages = 3
    else:
        # A800 / Data Center GPU optimization for massive batch sizes
        block_m = 256
        num_warps = 8
        num_stages = 3

    assert out_2d.shape == (M, N)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    kernel = (
        _w8a8_scaled_mm_nomask_kernel
        if _can_use_nomask_kernel(K, N)
        else _w8a8_scaled_mm_kernel
    )
    kernel[grid](
        xq_2d,
        qweight.qweight_t,
        scale_a,
        qweight.scale,
        out_2d,
        bias if bias is not None else out_2d,
        M,
        N,
        K,
        xq_2d.stride(0),
        xq_2d.stride(1),
        qweight.qweight_t.stride(0),
        qweight.qweight_t.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        STATIC_ACT=static_act,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_2d


def w8a8_scaled_mm(
    xq_2d: torch.Tensor,
    qweight: QuantizedWeight,
    scale_a: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype,
    static_act: bool,
) -> torch.Tensor:
    """Run only the quantized int8 GEMM + scale multiply stage."""
    assert xq_2d.ndim == 2
    K = xq_2d.shape[1]
    assert K == qweight.qweight_t.shape[0]

    M = xq_2d.shape[0]
    N = qweight.qweight_t.shape[1]
    out_2d = torch.empty((M, N), device=xq_2d.device, dtype=out_dtype)
    return w8a8_scaled_mm_out(
        xq_2d,
        qweight,
        scale_a,
        out_2d,
        bias=bias,
        static_act=static_act,
    )


def int8_mm_no_scale_out(
    xq_2d: torch.Tensor,
    qweight: QuantizedWeight,
    out_2d: torch.Tensor,
) -> torch.Tensor:
    assert xq_2d.ndim == 2
    K = xq_2d.shape[1]
    assert K == qweight.qweight_t.shape[0]

    M = xq_2d.shape[0]
    N = qweight.qweight_t.shape[1]
    assert out_2d.shape == (M, N)
    # --- Shape-based Heuristic Dispatcher ---
    block_n = min(128, triton.next_power_of_2(N))
    block_k = min(128, triton.next_power_of_2(K))
    if M < 2048:
        block_m = 32
        num_warps = 4
        num_stages = 3
    elif M < 4096:
        block_m = 128
        num_warps = 8
        num_stages = 3
    else:
        block_m = 256
        num_warps = 8
        num_stages = 3

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    kernel = (
        _int8_mm_no_scale_nomask_kernel
        if _can_use_nomask_kernel(K, N)
        else _int8_mm_no_scale_kernel
    )
    kernel[grid](
        xq_2d,
        qweight.qweight_t,
        out_2d,
        M,
        N,
        K,
        xq_2d.stride(0),
        xq_2d.stride(1),
        qweight.qweight_t.stride(0),
        qweight.qweight_t.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_2d


def int8_mm_no_scale(
    xq_2d: torch.Tensor,
    qweight: QuantizedWeight,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Run int8 GEMM without dequant scales, for timing the dot stage only."""
    assert xq_2d.ndim == 2
    K = xq_2d.shape[1]
    assert K == qweight.qweight_t.shape[0]

    M = xq_2d.shape[0]
    N = qweight.qweight_t.shape[1]
    out_2d = torch.empty((M, N), device=xq_2d.device, dtype=out_dtype)
    return int8_mm_no_scale_out(xq_2d, qweight, out_2d)


def w8a8_linear(
    x: torch.Tensor,
    qweight: QuantizedWeight,
    *,
    bias: torch.Tensor | None = None,
    act_mode: str,
    static_act_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run W8A8 Linear on x with arbitrary leading batch dimensions."""
    assert x.shape[-1] == qweight.qweight_t.shape[0]

    x_2d, original_batch_shape = flatten_input(x)
    xq, scale_a = quantize_activation(
        x_2d,
        act_mode=act_mode,
        static_act_scale=static_act_scale,
    )
    # Keep dequant scales fused into the Triton GEMM so the int32 accumulator
    # is never materialized as a separate tensor.
    out_2d = w8a8_scaled_mm(
        xq,
        qweight,
        scale_a,
        bias=bias,
        out_dtype=x.dtype,
        static_act=(act_mode == "static"),
    )
    return out_2d.view(*original_batch_shape, qweight.qweight_t.shape[1])
