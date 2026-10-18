"""Block-wise FP8 (E4M3) dequantization.

Drama 3 expert weights are stored as `torch.float8_e4m3fn` with a companion
`weight_scale_inv` tensor of dtype float32 holding one scale per
`weight_block_size = [128, 128]` block of the 2-D weight (`quant_method =
"bsgemm-moe"`, same layout as DeepSeek-V3 / vLLM block-FP8):

    weight[r, c] = fp8[r, c] * scale_inv[r // 128, c // 128]

Shapes in the checkpoint:  gate/up (512, 2048) -> scale (4, 16);
                           down    (2048, 512) -> scale (16, 4).
"""

from __future__ import annotations

import math

import torch

FP8_DTYPES = tuple(
    getattr(torch, n) for n in ("float8_e4m3fn", "float8_e4m3fnuz") if hasattr(torch, n)
)
E4M3_MAX = 448.0


def _check(weight: torch.Tensor, scale_inv: torch.Tensor, block: tuple[int, int]) -> tuple[int, int, int, int]:
    if weight.ndim < 2:
        raise ValueError(f"weight must be >= 2-D, got shape {tuple(weight.shape)}")
    if scale_inv.ndim != weight.ndim:
        raise ValueError(f"scale_inv ndim {scale_inv.ndim} != weight ndim {weight.ndim}")
    if weight.shape[:-2] != scale_inv.shape[:-2]:
        raise ValueError(f"leading dims differ: {tuple(weight.shape[:-2])} vs {tuple(scale_inv.shape[:-2])}")
    br, bc = int(block[0]), int(block[1])
    R, C = weight.shape[-2], weight.shape[-1]
    nbr, nbc = math.ceil(R / br), math.ceil(C / bc)
    if tuple(scale_inv.shape[-2:]) != (nbr, nbc):
        raise ValueError(
            f"scale_inv shape {tuple(scale_inv.shape[-2:])} does not match ceil({R}/{br}), ceil({C}/{bc}) = {(nbr, nbc)}"
        )
    return br, bc, nbr, nbc


def dequant_block_fp8(
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block: tuple[int, int] = (128, 128),
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a 2-D `[R, C]` or stacked `[..., R, C]` block-FP8 tensor.

    `weight_scale_inv` has shape `[..., ceil(R/br), ceil(C/bc)]`. Ragged last
    blocks (R or C not a multiple of the block) are handled by padding. Works on
    CPU and GPU; the multiply is done in float32 then cast to `out_dtype`.
    """
    if weight_fp8.dtype not in FP8_DTYPES and not weight_fp8.is_floating_point():
        raise TypeError(f"expected an fp8 (or float) weight, got {weight_fp8.dtype}")
    br, bc, nbr, nbc = _check(weight_fp8, weight_scale_inv, block)
    lead = tuple(weight_fp8.shape[:-2])
    R, C = weight_fp8.shape[-2], weight_fp8.shape[-1]

    w = weight_fp8.reshape(-1, R, C).to(torch.float32)
    s = weight_scale_inv.reshape(-1, nbr, nbc).to(torch.float32)
    pad_r, pad_c = nbr * br - R, nbc * bc - C
    if pad_r or pad_c:
        w = torch.nn.functional.pad(w, (0, pad_c, 0, pad_r))
    w = w.view(-1, nbr, br, nbc, bc) * s.view(-1, nbr, 1, nbc, 1)
    w = w.view(-1, nbr * br, nbc * bc)[:, :R, :C]
    return w.to(out_dtype).reshape(*lead, R, C)


def dequant_block_fp8_reference(
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block: tuple[int, int] = (128, 128),
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Slow, obviously-correct loop over blocks (tests only)."""
    br, bc, nbr, nbc = _check(weight_fp8, weight_scale_inv, block)
    lead = tuple(weight_fp8.shape[:-2])
    R, C = weight_fp8.shape[-2], weight_fp8.shape[-1]
    w = weight_fp8.reshape(-1, R, C).to(torch.float32)
    s = weight_scale_inv.reshape(-1, nbr, nbc).to(torch.float32)
    out = torch.empty_like(w)
    for e in range(w.shape[0]):
        for i in range(nbr):
            for j in range(nbc):
                rs, cs = slice(i * br, min((i + 1) * br, R)), slice(j * bc, min((j + 1) * bc, C))
                out[e, rs, cs] = w[e, rs, cs] * s[e, i, j]
    return out.to(out_dtype).reshape(*lead, R, C)


def quantize_block_fp8_reference(
    weight: torch.Tensor,
    block: tuple[int, int] = (128, 128),
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block absmax quantization (tests / fixtures only). Returns (fp8, scale_inv)
    such that `dequant_block_fp8(fp8, scale_inv) ~= weight`."""
    br, bc = int(block[0]), int(block[1])
    lead = tuple(weight.shape[:-2])
    R, C = weight.shape[-2], weight.shape[-1]
    nbr, nbc = math.ceil(R / br), math.ceil(C / bc)
    w = weight.reshape(-1, R, C).to(torch.float32)
    pad_r, pad_c = nbr * br - R, nbc * bc - C
    if pad_r or pad_c:
        w = torch.nn.functional.pad(w, (0, pad_c, 0, pad_r))
    wb = w.view(-1, nbr, br, nbc, bc)
    amax = wb.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12)  # [E, nbr, 1, nbc, 1]
    scale_inv = amax / E4M3_MAX
    q = (wb / scale_inv).clamp(-E4M3_MAX, E4M3_MAX).to(fp8_dtype)
    q = q.view(-1, nbr * br, nbc * bc)[:, :R, :C].reshape(*lead, R, C)
    return q.contiguous(), scale_inv.view(-1, nbr, nbc).reshape(*lead, nbr, nbc).contiguous()


__all__ = [
    "dequant_block_fp8",
    "dequant_block_fp8_reference",
    "quantize_block_fp8_reference",
    "FP8_DTYPES",
    "E4M3_MAX",
]
