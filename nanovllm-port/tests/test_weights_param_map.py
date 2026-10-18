"""Every tensor in the real drama-3 safetensors index must be classified and mapped."""

import json
import re
from collections import Counter

import pytest

from nanovllm_drama3.weights import param_map as pm


@pytest.fixture(scope="module")
def index():
    return pm.load_index()


@pytest.fixture(scope="module")
def entries(index):
    return pm.build_param_map(index)


def test_index_is_the_real_checkpoint(index):
    assert len(index) == 62245
    assert len(set(index.values())) == 8


def test_every_tensor_classified(entries, index):
    assert set(entries) == set(index)
    assert all(e.cls in pm.CLASSES for e in entries.values())


def test_class_counts(entries):
    counts = Counter(e.cls for e in entries.values())
    assert counts == {
        "backbone.embed_tokens": 1,
        "backbone.final_norm": 1,
        "backbone.layer_norm": 80,            # 2 x 40
        "backbone.full_attn": 60,             # 6 x 10 full-attention layers
        "backbone.linear_attn": 270,          # 9 x 30 GDN layers
        "backbone.moe_router": 40,
        "backbone.moe_shared_expert": 120,    # 3 x 40
        "backbone.moe_shared_expert_gate": 40,
        "backbone.moe_expert_fp8": 30720,     # 40 x 256 x 3
        "backbone.moe_expert_scale": 30720,
        "lm_head": 1,
        "tower.codebook_embeddings": 1,
        "tower.shared": 56,                   # 4 layers x 14
        "tower.residual": 126,                # 9 layers x 14
        "tower.heads": 9,
    }
    assert sum(counts.values()) == 62245


def test_backbone_targets_and_roles(entries):
    e = entries["model.language_model.layers.7.mlp.experts.3.up_proj.weight"]
    assert e.target == "model.layers.7.mlp.experts.3.gate_up_proj.weight"
    assert (e.role, e.shard_id, e.layer, e.expert) == (pm.ROLE_FP8_WEIGHT, 1, 7, 3)
    s = entries["model.language_model.layers.7.mlp.experts.3.up_proj.weight_scale_inv"]
    assert s.target == e.target and s.shard_id == 1 and s.role == pm.ROLE_FP8_SCALE
    assert s.scale_for == e.source
    d = entries["model.language_model.layers.7.mlp.experts.3.down_proj.weight"]
    assert d.target == "model.layers.7.mlp.experts.3.down_proj.weight" and d.shard_id is None
    sh = entries["model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"]
    assert sh.target == "model.layers.0.mlp.shared_expert.gate_up_proj.weight" and (sh.role, sh.shard_id) == (pm.ROLE_PACKED, 0)
    sg = entries["model.language_model.layers.0.mlp.shared_expert_gate.weight"]
    assert sg.target == "model.layers.0.mlp.shared_expert_gate.weight" and sg.role == pm.ROLE_WEIGHT
    assert entries["lm_head.weight"].target == "lm_head.weight"
    assert entries["model.language_model.embed_tokens.weight"].target == "model.embed_tokens.weight"
    assert entries["model.language_model.layers.0.linear_attn.A_log"].target == "model.layers.0.linear_attn.A_log"
    t = entries["audio_decoder_mtp.heads.4.weight"]
    assert t.role == pm.ROLE_SKIP and t.target == "tower.audio_decoder_mtp.heads.4.weight"
    assert entries["audio_decoder.codebook_embeddings.weight"].cls == "tower.codebook_embeddings"


def test_full_attention_only_on_every_fourth_layer(entries):
    attn_layers = sorted({e.layer for e in entries.values() if e.cls == "backbone.full_attn"})
    gdn_layers = sorted({e.layer for e in entries.values() if e.cls == "backbone.linear_attn"})
    assert attn_layers == [i for i in range(40) if i % 4 == 3]
    assert gdn_layers == [i for i in range(40) if i % 4 != 3]


def test_unknown_name_raises():
    with pytest.raises(pm.UnknownParam):
        pm.classify("model.language_model.layers.0.mlp.experts.0.gate_proj.qweight")
    with pytest.raises(pm.UnknownParam):
        pm.classify("model.visual.blocks.0.attn.qkv.weight")
    with pytest.raises(pm.UnknownParam):
        pm.build_param_map(["lm_head.weight", "mtp.layers.0.weight"])


def test_backbone_targets_exist_in_tiny_fusion_model(entries):
    """Targets for layers/experts that exist in the tiny model must be real parameter
    names of the fusion module (tiny config shares the 3 GDN + 1 full layer pattern)."""
    import torch

    from nanovllm_drama3.backbone import build_backbone
    from nanovllm_drama3.backbone.testing import tiny_text_config

    cfg = tiny_text_config()
    model = build_backbone(cfg, "cpu", torch.float32)
    names = set(dict(model.named_parameters()))
    checked = 0
    for e in entries.values():
        if e.role == pm.ROLE_SKIP:
            continue
        if e.layer is not None and e.layer >= cfg.num_hidden_layers:
            continue
        if e.expert is not None and e.expert >= cfg.num_experts:
            continue
        assert e.target in names, (e.source, e.target)
        checked += 1
    assert checked > 100
    # and every backbone parameter of the tiny model is produced by some checkpoint tensor
    targets = {e.target for e in entries.values() if e.role != pm.ROLE_SKIP}
    assert names <= targets


def test_param_map_json_is_current_and_complete(entries, tmp_path):
    """weights/param_map.json is the committed artefact; regenerating must reproduce it."""
    assert pm.PARAM_MAP_JSON.exists()
    committed = json.load(open(pm.PARAM_MAP_JSON))
    regenerated = pm.write_param_map_json(tmp_path / "pm.json")
    assert committed == regenerated
    assert committed["num_tensors"] == 62245
    assert sum(r["count"] for r in committed["patterns"]) == 62245
    # every pattern row carries the header dtype/shape
    assert all("shape" in r and "dtype" in r for r in committed["patterns"])
    fp8 = [r for r in committed["patterns"] if r["class"] == "backbone.moe_expert_fp8"]
    assert {r["dtype"] for r in fp8} == {"F8_E4M3"}
    assert {tuple(r["shape"]) for r in fp8} == {(512, 2048), (2048, 512)}
    scales = {tuple(r["shape"]) for r in committed["patterns"] if r["class"] == "backbone.moe_expert_scale"}
    assert scales == {(4, 16), (16, 4)}


def test_shapes_cover_index(index):
    shapes = pm.load_shapes()
    assert set(shapes) == set(index)
    dtypes = Counter(v["dtype"] for v in shapes.values())
    assert dtypes == {"BF16": 805, "F8_E4M3": 30720, "F32": 30720}
    non_bf16 = {k for k, v in shapes.items() if v["dtype"] != "BF16"}
    assert all(re.search(r"\.mlp\.experts\.\d+\.", k) for k in non_bf16)
