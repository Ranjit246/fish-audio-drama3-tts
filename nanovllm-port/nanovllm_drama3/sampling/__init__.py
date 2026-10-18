# SPDX-License-Identifier: Apache-2.0
"""Drama 3 sampling: main token with repetition-aware re-draw, codebook heads."""

from .sampler import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    GREEDY_TEMPERATURE,
    HEAD_SAMPLING,
    RAS_SAMPLING,
    RAS_TEMPERATURE,
    RAS_TOP_K,
    RAS_TOP_P,
    RAS_WINDOW,
    MainTokenDraw,
    MainTokenSampler,
    SamplingParams,
    main_candidate_ids,
    sample_codebook,
    sample_rows,
    sample_top_k_top_p,
)

__all__ = [
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_K",
    "DEFAULT_TOP_P",
    "GREEDY_TEMPERATURE",
    "HEAD_SAMPLING",
    "RAS_SAMPLING",
    "RAS_TEMPERATURE",
    "RAS_TOP_K",
    "RAS_TOP_P",
    "RAS_WINDOW",
    "MainTokenDraw",
    "MainTokenSampler",
    "SamplingParams",
    "main_candidate_ids",
    "sample_codebook",
    "sample_rows",
    "sample_top_k_top_p",
]
