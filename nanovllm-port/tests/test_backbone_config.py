"""Drama 3 config.json -> fusion text config (real bundled config)."""

import copy
import json

import pytest
import torch

from nanovllm_drama3.config import (
    BUNDLED_CONFIG,
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Drama3Config,
    FusionTextConfig,
    IncompatibleConfig,
    estimate_param_bytes,
    fusion_text_config_from_dict,
)


@pytest.fixture(scope="module")
def cfg() -> Drama3Config:
    return Drama3Config.bundled()


def test_bundled_config_is_real_drama3(cfg):
    raw = json.load(open(BUNDLED_CONFIG))
    assert raw["model_type"] == "fish_omni"
    assert raw["architectures"] == ["FishOmniModel"]
    assert cfg.text_config["model_type"] == "fish_omni_text"
    assert cfg.speech_template_version == "reference-bracket-v1"


def test_token_ids_and_codebooks(cfg):
    assert cfg.semantic_start_token_id == 248099
    assert cfg.semantic_end_token_id == 252194
    assert cfg.eos_token_id == 248046
    assert cfg.pad_token_id == 248044
    assert cfg.audio_pad_token_id == 248076
    assert cfg.num_codebooks == 10
    assert cfg.codebook_size == 4096
    rng = cfg.semantic_token_range
    assert len(rng) == 4096 and rng[0] == 248099 and rng[-1] == 252194
    rows = cfg.semantic_logit_rows()
    assert len(rows) == 4097 and rows[-1] == cfg.eos_token_id
    assert cfg.code0_to_token_id(0) == 248099
    assert cfg.code0_to_token_id(4095) == cfg.semantic_end_token_id


def test_quantization_fields(cfg):
    assert cfg.quant_method == "bsgemm-moe"
    assert cfg.weight_block_size == (128, 128)


def test_fusion_text_config_matches_known_facts(cfg):
    tc = cfg.fusion_text_config()
    assert isinstance(tc, FusionTextConfig)
    assert tc.model_type == "qwen3_5_moe"  # fusion _create_model dispatch key
    assert tc.dtype == torch.bfloat16
    assert tc.hidden_size == 2048
    assert tc.num_hidden_layers == 40 and len(tc.layer_types) == 40
    assert tc.num_full_attention_layers == 10 and tc.num_linear_attention_layers == 30
    # 3x linear then full, repeated
    for i, t in enumerate(tc.layer_types):
        assert t == (FULL_ATTENTION if i % 4 == 3 else LINEAR_ATTENTION)
    assert (tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim) == (16, 2, 256)
    assert (tc.linear_num_key_heads, tc.linear_num_value_heads) == (16, 32)
    assert (tc.linear_key_head_dim, tc.linear_value_head_dim, tc.linear_conv_kernel_dim) == (128, 128, 4)
    assert (tc.num_experts, tc.num_experts_per_tok, tc.moe_intermediate_size) == (256, 8, 512)
    assert tc.shared_expert_intermediate_size == 512
    assert tc.vocab_size == 252288
    assert tc.tie_word_embeddings is False
    assert tc.mlp_only_layers == [] and tc.decoder_sparse_step is None
    assert tc.rms_norm_eps == 1e-6
    assert tc.max_position_embeddings == 262144
    assert tc.rope_parameters["rope_theta"] == 1e7
    assert tc.rope_parameters["partial_rotary_factor"] == 0.25
    assert tc.rotary_dim == 64
    assert sum(tc.rope_parameters["mrope_section"]) == tc.rotary_dim // 2
    assert tc.norm_topk_prob is True


def test_fusion_config_shapes_match_checkpoint(cfg):
    """Derived shapes must equal the safetensors header shapes."""
    from nanovllm_drama3.weights.param_map import load_shapes

    tc = cfg.fusion_text_config()
    shapes = load_shapes()
    h = tc.hidden_size
    assert shapes["model.language_model.layers.3.self_attn.q_proj.weight"]["shape"] == [tc.num_attention_heads * tc.head_dim * 2, h]
    assert shapes["model.language_model.layers.3.self_attn.k_proj.weight"]["shape"] == [tc.num_key_value_heads * tc.head_dim, h]
    conv_dim = 2 * tc.linear_num_key_heads * tc.linear_key_head_dim + tc.linear_num_value_heads * tc.linear_value_head_dim
    assert shapes["model.language_model.layers.0.linear_attn.conv1d.weight"]["shape"] == [conv_dim, 1, tc.linear_conv_kernel_dim]
    assert shapes["model.language_model.layers.0.linear_attn.A_log"]["shape"] == [tc.linear_num_value_heads]
    assert shapes["model.language_model.layers.0.mlp.gate.weight"]["shape"] == [tc.num_experts, h]
    assert shapes["model.language_model.layers.0.mlp.shared_expert_gate.weight"]["shape"] == [1, h]
    assert shapes["model.language_model.layers.0.mlp.experts.0.gate_proj.weight"] == {"dtype": "F8_E4M3", "shape": [tc.moe_intermediate_size, h]}
    assert shapes["model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"] == {"dtype": "F32", "shape": [4, 16]}
    assert shapes["lm_head.weight"]["shape"] == [tc.vocab_size, h]


def test_to_dict_roundtrip(cfg):
    d = cfg.fusion_text_config().to_dict()
    assert d["dtype"] == "bfloat16"
    assert set(d) == set(FusionTextConfig._FIELDS)


@pytest.mark.parametrize(
    "key,value",
    [
        ("attn_output_gate", False),
        ("use_qk_norm", False),
        ("layernorm_zero_centered_gamma", False),
        ("rotary_interleaved", True),
        ("shared_expert_gate", False),
        ("hidden_act", "gelu"),
    ],
)
def test_incompatible_switches_raise(cfg, key, value):
    tc = copy.deepcopy(cfg.text_config)
    tc[key] = value
    with pytest.raises(IncompatibleConfig):
        fusion_text_config_from_dict(tc)


def test_unknown_layer_type_and_rope_scaling_raise(cfg):
    tc = copy.deepcopy(cfg.text_config)
    tc["layer_types"][0] = "sliding_attention"
    with pytest.raises(IncompatibleConfig):
        fusion_text_config_from_dict(tc)
    tc = copy.deepcopy(cfg.text_config)
    tc["rope_parameters"]["rope_type"] = "yarn"
    with pytest.raises(IncompatibleConfig):
        fusion_text_config_from_dict(tc)


def test_mlp_only_layers_translation(cfg):
    tc = copy.deepcopy(cfg.text_config)
    tc["mlp_layer_types"][5] = "mlp"
    out = fusion_text_config_from_dict(tc)
    assert out.mlp_only_layers == [5]


def test_wrong_model_type_rejected(cfg):
    raw = copy.deepcopy(cfg.raw)
    raw["model_type"] = "qwen3_5_moe"
    with pytest.raises(IncompatibleConfig):
        Drama3Config.from_dict(raw)


def test_param_budget_estimate(cfg):
    est = estimate_param_bytes(cfg.fusion_text_config())
    # 40 layers x 256 experts x 3 x 2048 x 512 x 2 bytes ~= 64 GiB
    assert 60e9 < est["experts"] < 70e9
    assert est["lm_head_semantic_slice"] < est["lm_head"] / 50
