# SPDX-License-Identifier: Apache-2.0
"""Offline/static preprocessing helpers for W8A8 INT8 Linear."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

# (K, N): X[M, K] @ W[N, K].T -> Y[M, N]
LINEAR_KN_SHAPES = [
    (512, 1536),  # Transformer qkv_proj
    (512, 512),  # Transformer out_proj
    (512, 1024),  # MoE expert fc1
    (1024, 512),  # MoE expert fc2
    (14336, 512),  # RepEncoder projection
]


@dataclass
class QuantizedWeight:
    # Stored transposed for GEMM: [K, N].
    qweight_t: torch.Tensor
    # Per-output-channel scale: [N].
    scale: torch.Tensor
    # Kept for bf16 reference comparison in benchmarks.
    fp_weight: torch.Tensor

    def to(self, device: torch.device | str) -> "QuantizedWeight":
        return QuantizedWeight(
            qweight_t=self.qweight_t.to(device),
            scale=self.scale.to(device),
            fp_weight=self.fp_weight.to(device),
        )


@dataclass
class LinearArtifact:
    K: int
    N: int
    qweight: QuantizedWeight
    static_act_scale: torch.Tensor


def quantize_weight_per_channel(weight: torch.Tensor) -> QuantizedWeight:
    """Offline/static symmetric int8 weight quantization for W[N, K]."""
    assert weight.ndim == 2
    amax = weight.float().abs().amax(dim=1).clamp(min=1e-8)
    scale = amax / 127.0
    qweight = torch.round(weight.float() / scale[:, None]).clamp(-128, 127)
    qweight = qweight.to(torch.int8)
    return QuantizedWeight(
        qweight_t=qweight.t(),  # 保持 column-major 连续性以满足 cuBLAS Lt 需求
        scale=scale.contiguous(),
        fp_weight=weight.contiguous(),
    )


def choose_act_static_scale(calib_x: torch.Tensor) -> torch.Tensor:
    """Simulate calibration-time static per-tensor activation scale."""
    scale = calib_x.float().abs().amax().clamp(min=1e-8) / 127.0
    return scale.reshape(1).contiguous()


def artifact_path(artifact_dir: Path, K: int, N: int) -> Path:
    return artifact_dir / f"linear_K={K}_N={N}.pt"


def prepare_artifacts(
    artifact_dir: Path,
    *,
    dtype: torch.dtype,
    calib_tokens: int,
    seed: int,
) -> None:
    """Create random quantized artifacts that mimic a W8A8 checkpoint."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for K, N in LINEAR_KN_SHAPES:
        weight = torch.randn((N, K), generator=generator, dtype=dtype) / (K**0.5)
        qweight = quantize_weight_per_channel(weight)
        calib_x = torch.randn((calib_tokens, K), generator=generator, dtype=dtype)
        static_act_scale = choose_act_static_scale(calib_x)

        path = artifact_path(artifact_dir, K, N)
        torch.save(
            {
                "K": K,
                "N": N,
                "qweight_t": qweight.qweight_t.cpu(),
                "weight_scale": qweight.scale.cpu(),
                "fp_weight": qweight.fp_weight.cpu(),
                "static_act_scale": static_act_scale.cpu(),
            },
            path,
        )
        print(
            f"saved {path} "
            f"qweight_t={tuple(qweight.qweight_t.shape)} "
            f"weight_scale={tuple(qweight.scale.shape)} "
            f"static_act_scale={tuple(static_act_scale.shape)}"
        )


def load_artifact(artifact_dir: Path, K: int, N: int, device: str) -> LinearArtifact:
    data = torch.load(artifact_path(artifact_dir, K, N), map_location=device)
    qweight = QuantizedWeight(
        # 强制将读取的张量转换为 column-major 连续，满足 cuBLAS Tensor Core 苛刻的内存对齐需求
        qweight_t=data["qweight_t"].t().contiguous().t(),
        scale=data["weight_scale"].contiguous(),
        fp_weight=data["fp_weight"].contiguous(),
    )
    return LinearArtifact(
        K=int(data["K"]),
        N=int(data["N"]),
        qweight=qweight,
        static_act_scale=data["static_act_scale"].contiguous(),
    )


def make_input(
    batch_size: int,
    seq_len: int | None,
    K: int,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create [B, K] or [B, T, K] input.

    Triton GEMM still consumes 2D tensors. The kernel wrapper flattens all
    leading dimensions into M = prod(x.shape[:-1]).
    """
    if seq_len is None:
        return torch.randn((batch_size, K), device=device, dtype=dtype)
    return torch.randn((batch_size, seq_len, K), device=device, dtype=dtype)
