# SPDX-License-Identifier: Apache-2.0
"""Codebook-tower hyper-parameters under the checkpoint's own field names.

Mirrors ``FishOmniAudioDecoderConfig`` in the vLLM-Omni reference
(``_reference/configuration_fish_omni.py``) but as a plain dataclass with no
transformers dependency. ``TowerConfig.drama3()`` holds the released
checkpoint's values (``audio_decoder_config`` in ``config.json``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TowerConfig:
    dim: int = 2048
    n_head: int = 16
    n_local_heads: int = 2
    head_dim: int = 256
    intermediate_size: int = 6144
    norm_eps: float = 1e-6
    num_codebooks: int = 10
    vocab_size: int = 4096
    shared_enrichment_layers: int = 4
    rope_base: float = 1e7
    rotary_percent: float = 0.25
    vq_backbone_codebooks: int = 1

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rotary_percent)

    @property
    def num_residual_layers(self) -> int:
        """One residual layer (and head) per codebook after code 0."""
        return self.num_codebooks - 1

    @property
    def num_layers(self) -> int:
        return self.shared_enrichment_layers + self.num_residual_layers

    @classmethod
    def drama3(cls) -> "TowerConfig":
        return cls()

    @classmethod
    def from_hf_dict(cls, audio_decoder_config: Mapping[str, Any]) -> "TowerConfig":
        """Build from ``config.json["audio_decoder_config"]``; unknown keys are ignored."""
        fields = cls.__dataclass_fields__
        kwargs = {k: v for k, v in audio_decoder_config.items() if k in fields}
        return cls(**kwargs)

    @classmethod
    def tiny(cls, **overrides: Any) -> "TowerConfig":
        """A small random-weight configuration for CPU tests; keeps the real layer
        counts so the parameter-name set matches the checkpoint."""
        base = dict(
            dim=32,
            n_head=4,
            n_local_heads=2,
            head_dim=8,
            intermediate_size=64,
            norm_eps=1e-6,
            num_codebooks=10,
            vocab_size=16,
            shared_enrichment_layers=4,
            rope_base=1e4,
            rotary_percent=0.25,
            vq_backbone_codebooks=1,
        )
        base.update(overrides)
        return cls(**base)
