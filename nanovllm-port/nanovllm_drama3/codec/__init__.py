# SPDX-License-Identifier: Apache-2.0
"""Fish Speech S2 Pro DAC codec (vendored from vllm-omni c470395) for the Drama 3 port.

Depends only on torch and numpy. See ``VENDORED-wp3.md`` for provenance and
``streaming_notes.md`` for what a streaming decoder would need.
"""

from .dac_utils import (
    DAC_HOP_LENGTH,
    DAC_NUM_CODEBOOKS,
    DAC_RESIDUAL_CODEBOOK_SIZE,
    DAC_SAMPLE_RATE,
    DAC_SEMANTIC_CODEBOOK_SIZE,
    bake_weight_norm,
    build_dac_codec,
    dac_codebook_sizes,
)
from .fish_dac import FishDAC, LoadReport, to_mono_tensor
from .resample import resample

__all__ = [
    "DAC_HOP_LENGTH",
    "DAC_NUM_CODEBOOKS",
    "DAC_RESIDUAL_CODEBOOK_SIZE",
    "DAC_SAMPLE_RATE",
    "DAC_SEMANTIC_CODEBOOK_SIZE",
    "FishDAC",
    "LoadReport",
    "bake_weight_norm",
    "build_dac_codec",
    "dac_codebook_sizes",
    "resample",
    "to_mono_tensor",
]
