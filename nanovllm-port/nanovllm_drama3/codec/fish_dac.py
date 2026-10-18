# SPDX-License-Identifier: Apache-2.0
# Encode/decode wrapper around the vendored Fish Speech S2 Pro DAC codec.
# The encode path follows vllm-omni c470395 fish_speech/dac_encoder.py
# (``_load_dac_codec``, ``_prepare_reference_audio_tensor``,
# ``encode_reference_audio_codes``); the decode path follows the module
# (non-streaming) branch of fish_speech/fish_speech_dac_decoder.py
# (``_ensure_codec_loaded`` + ``_decode_codes``).

"""``FishDAC``: load ``codec.pth``, encode reference audio to codes, decode codes to 44.1 kHz audio.

Layouts
-------
* codes are ``LongTensor[frames, 10]`` (frame-major; column 0 is the semantic
  codebook, columns 1-9 the residual ones). The DAC modules themselves use
  ``[batch, 10, frames]``; the transpose happens inside this class.
* ``decode`` returns ``FloatTensor[frames * hop_length]`` (mono, float32 in
  ``[-1, 1]``, 44.1 kHz). ``hop_length`` is 2048 samples, so the frame rate is
  44100 / 2048 = 21.533 Hz (46.44 ms per frame).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .dac_utils import (
    DAC_HOP_LENGTH,
    DAC_NUM_CODEBOOKS,
    DAC_SAMPLE_RATE,
    bake_weight_norm,
    build_dac_codec,
    dac_codebook_sizes,
)
from .resample import resample

ArrayLike = np.ndarray | torch.Tensor | list[float]


@dataclass
class LoadReport:
    """What ``FishDAC.load`` found in the checkpoint."""

    codec_path: str
    num_state_keys: int
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    baked_weight_norms: int = 0
    num_parameters: int = 0


def to_mono_tensor(wav: ArrayLike, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Mono ``[samples]`` tensor; accepts ``[samples]``, ``[channels, samples]`` or ``[samples, channels]``.

    Same heuristics as the reference ``_prepare_reference_audio_tensor``: a 2-D
    input whose first dim is <= 8 and smaller than the second is channels-first.
    """
    if isinstance(wav, torch.Tensor):
        t = wav.detach()
    else:
        t = torch.as_tensor(np.asarray(wav))
    t = t.to(device=device, dtype=dtype)
    if t.ndim == 2:
        if t.shape[0] <= 8 and t.shape[1] > t.shape[0]:
            t = t.mean(dim=0)
        elif t.shape[-1] <= 8 and t.shape[0] > t.shape[-1]:
            t = t.mean(dim=-1)
        else:
            t = t.mean(dim=0)
    elif t.ndim > 2:
        t = t.reshape(-1, t.shape[-1]).mean(dim=0)
    return t.flatten()


class FishDAC:
    """Fish Speech S2 Pro DAC codec (encoder + quantizer + decoder) on one device/dtype.

    Use :meth:`load`; the constructor takes an already-built ``DAC`` module.
    """

    sample_rate: int = DAC_SAMPLE_RATE
    hop_length: int = DAC_HOP_LENGTH
    num_codebooks: int = DAC_NUM_CODEBOOKS
    codebook_sizes: tuple[int, ...] = dac_codebook_sizes()

    def __init__(self, codec: nn.Module, *, device: torch.device, dtype: torch.dtype, report: LoadReport | None = None):
        self.codec = codec
        self.device = device
        self.dtype = dtype
        self.report = report
        # The DAC module reports hop = prod(encoder_rates) = 512 and frame_length = hop * 4
        # (the quantizer downsamples a further 2*2); a codec frame is frame_length samples.
        assert int(codec.frame_length) == self.hop_length, (codec.frame_length, self.hop_length)
        assert int(codec.sample_rate) == self.sample_rate

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(
        cls,
        codec_path: str | os.PathLike,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        *,
        bake: bool = True,
    ) -> FishDAC:
        """Build the S2 Pro DAC, load ``codec.pth`` (a plain state dict, or one under ``"generator"``),
        fold weight norms into plain weights, move to ``device``/``dtype``, eval mode.

        ``dtype`` applies to the whole codec. Encoding should run in float32 (as the
        reference does); bf16/fp16 are meant for decode on GPU.
        """
        codec_path = os.fspath(codec_path)
        if os.path.isdir(codec_path):
            codec_path = os.path.join(codec_path, "codec.pth")
        if not os.path.exists(codec_path):
            raise FileNotFoundError(codec_path)
        device = torch.device(device)

        codec = build_dac_codec()
        state_dict = torch.load(codec_path, map_location="cpu", weights_only=True)
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        result = codec.load_state_dict(state_dict, strict=False)
        report = LoadReport(
            codec_path=codec_path,
            num_state_keys=len(state_dict),
            missing_keys=list(result.missing_keys),
            unexpected_keys=list(result.unexpected_keys),
        )
        if bake:
            report.baked_weight_norms = bake_weight_norm(codec)
        codec = codec.to(device=device, dtype=dtype)
        codec.eval()
        for p in codec.parameters():
            p.requires_grad_(False)
        report.num_parameters = sum(p.numel() for p in codec.parameters())
        return cls(codec, device=device, dtype=dtype, report=report)

    # ----------------------------------------------------------------- helpers
    @property
    def frame_rate(self) -> float:
        """Codec frames per second: 44100 / 2048 = 21.53 Hz."""
        return self.sample_rate / self.hop_length

    def num_frames(self, num_samples: int, sample_rate: int | None = None) -> int:
        """Frames ``encode_reference`` yields for ``num_samples`` at ``sample_rate`` (default 44.1 kHz)."""
        sr = self.sample_rate if sample_rate is None else int(sample_rate)
        if sr != self.sample_rate:
            g = math.gcd(sr, self.sample_rate)
            num_samples = math.ceil((self.sample_rate // g) * num_samples / (sr // g))
        return math.ceil(num_samples / self.hop_length)

    def prepare_reference(self, wav: ArrayLike, sample_rate: int) -> torch.Tensor:
        """Mono float waveform at 44.1 kHz on the codec device (the exact tensor that gets encoded)."""
        t = to_mono_tensor(wav, device=self.device, dtype=torch.float32)
        if int(sample_rate) != self.sample_rate:
            t = resample(t.unsqueeze(0), int(sample_rate), self.sample_rate).squeeze(0)
        return t.to(self.dtype)

    # ------------------------------------------------------------------ encode
    @torch.inference_mode()
    def encode_reference(self, wav: ArrayLike, sample_rate: int) -> torch.Tensor:
        """Encode reference audio to ``LongTensor[frames, 10]`` (on the codec device).

        The waveform is mixed to mono, resampled to 44.1 kHz, right-padded to a
        multiple of ``hop_length`` and encoded in one pass.
        """
        if self.codec.encoder is None:
            raise RuntimeError("codec was loaded without its encoder")
        audio = self.prepare_reference(wav, sample_rate)
        if audio.numel() == 0:
            return torch.empty((0, self.num_codebooks), dtype=torch.long, device=self.device)
        audio = audio[None, None, :]  # [1, 1, T]
        lengths = torch.tensor([audio.shape[-1]], device=self.device, dtype=torch.long)
        codes, _ = self.codec.encode(audio, lengths)  # [1, 10, frames]
        return codes[0].transpose(0, 1).to(torch.long).contiguous()

    # ------------------------------------------------------------------ decode
    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode ``LongTensor[frames, 10]`` to a mono float32 waveform ``[frames * hop_length]`` at 44.1 kHz.

        Codes are clamped into each codebook's range (as the reference does), so an
        EOS or out-of-range value in column 0 decodes as the last semantic entry
        rather than crashing; strip EOS frames before calling.
        """
        if self.codec.decoder is None:
            raise RuntimeError("codec was loaded without its decoder")
        if codes.ndim != 2 or codes.shape[1] != self.num_codebooks:
            raise ValueError(f"expected codes [frames, {self.num_codebooks}], got {tuple(codes.shape)}")
        frames = int(codes.shape[0])
        if frames == 0:
            return torch.empty((0,), dtype=torch.float32, device=self.device)
        indices = codes.to(device=self.device, dtype=torch.long).transpose(0, 1)[None]  # [1, 10, frames]
        lengths = torch.tensor([frames], device=self.device, dtype=torch.long)
        wav, audio_lengths = self.codec.decode(indices, lengths)  # [1, 1, T]
        n = int(audio_lengths[0])
        return wav[0, 0, :n].to(torch.float32)

    def decode_batch(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode several ``[frames_i, 10]`` sequences in one padded batch (right-padded with zeros)."""
        if not codes_list:
            return []
        frames = [int(c.shape[0]) for c in codes_list]
        batch = torch.zeros((len(codes_list), self.num_codebooks, max(frames)), dtype=torch.long, device=self.device)
        for i, c in enumerate(codes_list):
            batch[i, :, : frames[i]] = c.to(device=self.device, dtype=torch.long).transpose(0, 1)
        with torch.inference_mode():
            wav, _ = self.codec.decode(batch, torch.tensor(frames, device=self.device))
        return [wav[i, 0, : frames[i] * self.hop_length].to(torch.float32) for i in range(len(codes_list))]
