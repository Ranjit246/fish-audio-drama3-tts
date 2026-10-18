"""Block-FP8 dequantization vs a slow reference loop (CPU)."""

import pytest
import torch

from nanovllm_drama3.weights.fp8 import (
    E4M3_MAX,
    dequant_block_fp8,
    dequant_block_fp8_reference,
    quantize_block_fp8_reference,
)


def _rand(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * 0.02


@pytest.mark.parametrize(
    "shape,block",
    [
        ((512, 2048), (128, 128)),   # gate/up proj, scale (4, 16)
        ((2048, 512), (128, 128)),   # down proj, scale (16, 4)
        ((40, 24), (8, 8)),          # small, exact multiple
        ((37, 50), (16, 8)),         # ragged blocks (pad path)
        ((3, 40, 24), (8, 8)),       # stacked experts
        ((2, 37, 50), (16, 8)),      # stacked + ragged
    ],
)
def test_matches_reference_loop(shape, block):
    w = _rand(shape)
    q, s = quantize_block_fp8_reference(w, block)
    assert q.dtype == torch.float8_e4m3fn and s.dtype == torch.float32
    import math

    assert tuple(s.shape) == tuple(shape[:-2]) + (math.ceil(shape[-2] / block[0]), math.ceil(shape[-1] / block[1]))
    fast = dequant_block_fp8(q, s, block)
    slow = dequant_block_fp8_reference(q, s, block)
    assert fast.dtype == torch.bfloat16 and fast.shape == w.shape
    assert torch.equal(fast, slow)
    # Also faithful to the original within E4M3 precision (3 mantissa bits -> ~6%).
    err = (fast.float() - w).abs()
    assert (err <= w.abs() * 0.07 + 1e-6).all()


def test_real_checkpoint_scale_shapes():
    assert dequant_block_fp8(torch.zeros(512, 2048, dtype=torch.float8_e4m3fn), torch.ones(4, 16)).shape == (512, 2048)
    assert dequant_block_fp8(torch.zeros(2048, 512, dtype=torch.float8_e4m3fn), torch.ones(16, 4)).shape == (2048, 512)


def test_scale_is_applied_per_block():
    """Each block multiplied by its own scale: use a ones weight and distinct scales."""
    w = torch.ones(16, 24, dtype=torch.float8_e4m3fn)
    s = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    out = dequant_block_fp8(w, s, (8, 8), out_dtype=torch.float32)
    for i in range(2):
        for j in range(3):
            blk = out[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            assert torch.equal(blk, torch.full((8, 8), float(s[i, j])))


def test_out_dtype_and_float32_math():
    w = torch.full((4, 4), 2.0).to(torch.float8_e4m3fn)
    s = torch.tensor([[0.125]])
    out = dequant_block_fp8(w, s, (4, 4), out_dtype=torch.float32)
    assert out.dtype == torch.float32 and torch.equal(out, torch.full((4, 4), 0.25))


def test_bad_scale_shape_raises():
    w = torch.zeros(512, 2048, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        dequant_block_fp8(w, torch.ones(16, 4))
    with pytest.raises(ValueError):
        dequant_block_fp8(w, torch.ones(4, 16, 1))
    with pytest.raises(ValueError):
        dequant_block_fp8(torch.zeros(512, dtype=torch.float8_e4m3fn), torch.ones(4))


def test_quantize_reference_saturates():
    w = torch.tensor([[1000.0, -1000.0], [0.0, 1.0]])
    q, s = quantize_block_fp8_reference(w, (2, 2))
    assert float(q.float().abs().max()) <= E4M3_MAX
    assert torch.allclose(dequant_block_fp8(q, s, (2, 2), out_dtype=torch.float32), w, rtol=0.07, atol=1e-3)
