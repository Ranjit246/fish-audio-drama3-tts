# SPDX-License-Identifier: BSD-2-Clause
# Port of ``torchaudio.functional.resample`` (torchaudio, BSD-2-Clause,
# https://github.com/pytorch/audio) restricted to the ``sinc_interp_hann``
# method with torchaudio's defaults (lowpass_filter_width=6, rolloff=0.99).
#
# vllm-omni's reference encoder (fish_speech/dac_encoder.py) resamples with
# ``torchaudio.transforms.Resample(sr, 44100)``; this file reproduces that
# kernel bit-for-bit (the test suite checks it against torchaudio when
# torchaudio is importable) so the codec package depends only on torch.

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F

LOWPASS_FILTER_WIDTH = 6
ROLLOFF = 0.99


@lru_cache(maxsize=16)
def _sinc_kernel(orig_freq: int, new_freq: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    """Windowed-sinc polyphase kernel ``[new_freq, 1, 2*width + orig_freq]`` (frequencies already gcd-reduced).

    Matches ``torchaudio.transforms.Resample`` (``dtype=None``) bit-for-bit: ``idx`` in float64, the
    per-phase offset ``arange(0, -new_freq, -1) / new_freq`` in torch's default float32 (an int64
    ``arange`` divided by an int), their sum in float64, and the kernel cast to float32 at the end.
    """
    base_freq = min(orig_freq, new_freq) * ROLLOFF
    width = math.ceil(LOWPASS_FILTER_WIDTH * orig_freq / base_freq)
    idx = torch.arange(-width, width + orig_freq, dtype=torch.float64)[None, None] / orig_freq
    phase = torch.arange(0, -new_freq, -1, dtype=torch.int64)[:, None, None] / new_freq  # float32, as torchaudio
    t = phase + idx
    t *= base_freq
    t = t.clamp_(-LOWPASS_FILTER_WIDTH, LOWPASS_FILTER_WIDTH)
    window = torch.cos(t * math.pi / LOWPASS_FILTER_WIDTH / 2) ** 2
    t *= math.pi
    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0, dtype=t.dtype), t.sin() / t)
    kernels *= window * scale
    return kernels.to(dtype=torch.float32).to(device=device, dtype=dtype), width


def resample(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """Resample ``waveform [..., time]`` from ``orig_freq`` to ``new_freq`` Hz (torchaudio-equivalent)."""
    orig_freq, new_freq = int(orig_freq), int(new_freq)
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError(f"sample rates must be positive, got {orig_freq} -> {new_freq}")
    if orig_freq == new_freq:
        return waveform
    gcd = math.gcd(orig_freq, new_freq)
    orig_freq //= gcd
    new_freq //= gcd
    kernel, width = _sinc_kernel(orig_freq, new_freq, waveform.device, waveform.dtype)

    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])
    num_wavs, length = waveform.shape
    waveform = F.pad(waveform, (width, width + orig_freq))
    resampled = F.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = math.ceil(new_freq * length / orig_freq)
    resampled = resampled[..., :target_length]
    return resampled.view(shape[:-1] + resampled.shape[-1:])
