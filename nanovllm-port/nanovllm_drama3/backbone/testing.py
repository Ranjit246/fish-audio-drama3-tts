"""Tiny configs for CPU tests. Shapes keep the real structural ratios
(3 GDN : 1 full-attention, gated q_proj, GQA ratio 2 in GDN) at toy sizes."""

from __future__ import annotations

import copy
from typing import Any

from nanovllm_drama3.config import BUNDLED_CONFIG, Drama3Config, FusionTextConfig

TINY_TEXT = dict(
    hidden_size=64,
    num_hidden_layers=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    mlp_layer_types=["moe"] * 4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    intermediate_size=32,
    vocab_size=160,
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
    rope_parameters={"partial_rotary_factor": 0.25, "rope_theta": 10_000.0, "rope_type": "default"},
)

# Token ids chosen so semantic range (8 codes) + eos fit in the toy vocab.
TINY_TOP = dict(
    semantic_start_token_id=100,
    semantic_end_token_id=107,
    eos_token_id=90,
    pad_token_id=91,
    audio_pad_token_id=92,
)


def tiny_drama3_config_dict() -> dict[str, Any]:
    """Real Drama 3 config.json with every size shrunk; keeps all switches."""
    raw = copy.deepcopy(Drama3Config.from_file(BUNDLED_CONFIG).raw)
    raw.update(TINY_TOP)
    raw["text_config"].update(copy.deepcopy(TINY_TEXT))
    raw["text_config"]["eos_token_id"] = TINY_TOP["eos_token_id"]
    raw["text_config"]["pad_token_id"] = TINY_TOP["pad_token_id"]
    ad = raw["audio_decoder_config"]
    ad.update(dict(vocab_size=8, num_codebooks=3, dim=64, text_dim=64, head_dim=32, n_head=2, n_local_heads=1,
                   intermediate_size=96, n_layer=2, shared_enrichment_layers=1))
    return raw


def tiny_drama3_config() -> Drama3Config:
    return Drama3Config.from_dict(tiny_drama3_config_dict())


def tiny_text_config(**overrides: Any) -> FusionTextConfig:
    import torch

    cfg = tiny_drama3_config().fusion_text_config(dtype=overrides.pop("dtype", torch.float32))
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
