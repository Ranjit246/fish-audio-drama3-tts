"""load_backbone() smoke test: tiny randomly-initialised checkpoint written in the
drama-3 naming scheme (FP8 experts + block scales, tower tensors, 2 shards)."""

import json

import pytest
import torch
from safetensors.torch import save_file

from nanovllm_drama3.backbone import build_backbone, gdn_layers
from nanovllm_drama3.backbone.testing import tiny_drama3_config_dict
from nanovllm_drama3.config import Drama3Config
from nanovllm_drama3.weights import param_map as pm
from nanovllm_drama3.weights.fp8 import dequant_block_fp8, quantize_block_fp8_reference
from nanovllm_drama3.weights.loader import LoadError, load_backbone

BLOCK = (16, 16)  # tiny analogue of the real 128x128 (hidden 64 / moe 32 -> scales (2,4)/(4,2))


def _write_tiny_checkpoint(tmp_path, dtype=torch.bfloat16, seed=0):
    """Returns (model_dir, reference_model, drama_cfg)."""
    raw = tiny_drama3_config_dict()
    raw["quantization_config"]["weight_block_size"] = list(BLOCK)
    raw["dtype"] = "bfloat16"
    cfg = Drama3Config.from_dict(raw)
    tc = cfg.fusion_text_config(dtype=dtype)

    torch.manual_seed(seed)
    ref = build_backbone(tc, "cpu", dtype)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_((torch.randn(p.shape, generator=g) * 0.05).to(p.dtype))
        for layer in gdn_layers(ref):
            # The checkpoint stores A_log / norm.weight in bf16 while the fusion params
            # are float32: round the reference so loaded == reference exactly.
            layer.A_log.copy_(torch.rand(layer.A_log.shape, generator=g).log().to(torch.bfloat16).float())
            layer.norm.weight.copy_(layer.norm.weight.to(torch.bfloat16).float())

    # Reverse the param map: fusion param -> drama-3 tensors.
    tensors: dict[str, torch.Tensor] = {}
    params = dict(ref.named_parameters())
    inter = tc.moe_intermediate_size
    for name, p in params.items():
        t = p.detach().clone()
        if name == "lm_head.weight":
            tensors["lm_head.weight"] = t
            continue
        src = pm.BACKBONE_PREFIX + name[len("model."):]
        if name.endswith(".gate_up_proj.weight"):
            base = src[: -len(".gate_up_proj.weight")]
            parts = {"gate_proj": t[:inter], "up_proj": t[inter:]}
            if ".experts." in name:
                for proj, w in parts.items():
                    q, s = quantize_block_fp8_reference(w.float(), BLOCK)
                    tensors[f"{base}.{proj}.weight"] = q
                    tensors[f"{base}.{proj}.weight_scale_inv"] = s
                    with torch.no_grad():  # reference holds the dequantized value
                        (p[:inter] if proj == "gate_proj" else p[inter:]).copy_(dequant_block_fp8(q, s, BLOCK, dtype))
            else:
                for proj, w in parts.items():
                    tensors[f"{base}.{proj}.weight"] = w.contiguous()
        elif ".experts." in name and name.endswith(".down_proj.weight"):
            q, s = quantize_block_fp8_reference(t.float(), BLOCK)
            tensors[src] = q
            tensors[src + "_scale_inv"] = s
            with torch.no_grad():
                p.copy_(dequant_block_fp8(q, s, BLOCK, dtype))
        elif name.endswith("linear_attn.A_log") or name.endswith("linear_attn.norm.weight"):
            tensors[src] = t.to(torch.bfloat16)  # checkpoint stores these in bf16
        else:
            tensors[src] = t

    # Tower tensors the backbone loader must skip (names from the real index).
    h = tc.hidden_size
    tensors["audio_decoder.codebook_embeddings.weight"] = torch.zeros(3 * 8, h, dtype=dtype)
    tensors["audio_decoder_mtp_shared.layers.0.semantic_embedding.weight"] = torch.zeros(8, h, dtype=dtype)
    tensors["audio_decoder_mtp.layers.0.transformer_layer.attention.wqkv.weight"] = torch.zeros(4 * h, h, dtype=dtype)
    tensors["audio_decoder_mtp.heads.0.weight"] = torch.zeros(8, h, dtype=dtype)

    for n in list(tensors):
        assert pm.classify(n)  # every emitted name must be a known drama-3 name

    model_dir = tmp_path / "tiny-drama-3"
    model_dir.mkdir()
    names = sorted(tensors)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    weight_map = {}
    for fname, keys in shards.items():
        save_file({k: tensors[k].contiguous() for k in keys}, str(model_dir / fname))
        weight_map.update({k: fname for k in keys})
    json.dump({"metadata": {}, "weight_map": weight_map}, open(model_dir / "model.safetensors.index.json", "w"))
    json.dump(raw, open(model_dir / "config.json", "w"))
    return model_dir, ref, cfg


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    return _write_tiny_checkpoint(tmp_path_factory.mktemp("ckpt"))


def test_load_backbone_matches_reference(tiny_ckpt):
    model_dir, ref, cfg = tiny_ckpt
    model, report = load_backbone(model_dir, device="cpu", dtype=torch.bfloat16)
    assert report.missing == [] and report.unexpected == []
    assert len(report.shards) == 2
    n_layers, n_experts = cfg.text_config["num_hidden_layers"], cfg.text_config["num_experts"]
    assert report.dequantized_experts == n_layers * n_experts * 3
    assert report.loaded_by_class["backbone.moe_expert_fp8"] == n_layers * n_experts * 3
    assert report.loaded_by_class["backbone.moe_shared_expert_gate"] == n_layers
    assert report.loaded_by_class["lm_head"] == 1
    assert dict(report.skipped_tower) == {
        "tower.codebook_embeddings": 1, "tower.shared": 1, "tower.residual": 1, "tower.heads": 1,
    }
    got = dict(model.named_parameters())
    for name, p in ref.named_parameters():
        assert got[name].dtype == p.dtype, name
        assert torch.equal(got[name], p), name
    # bf16 checkpoint tensors landed in the float32 GDN params via copy_ (cast)
    assert got["model.layers.0.linear_attn.A_log"].dtype == torch.float32
    print(report.summary())


def test_loaded_model_runs_and_matches_reference_forward(tiny_ckpt):
    model_dir, ref, cfg = tiny_ckpt
    model, _ = load_backbone(model_dir, device="cpu", dtype=torch.bfloat16)
    from nanovllm.utils.context import reset_context, set_context

    ids = torch.randint(0, cfg.fusion_text_config().vocab_size, (6,))
    cu = torch.tensor([0, 6], dtype=torch.int32)
    out = []
    for m in (ref, model):
        set_context(True, cu, cu, 6, 6, torch.arange(6, dtype=torch.int32), None, None, state_indices=None)
        with torch.inference_mode():
            out.append(m.compute_logits(m(ids, torch.arange(6))))
        reset_context()
    assert out[0].shape == (1, cfg.fusion_text_config().vocab_size)
    torch.testing.assert_close(out[0], out[1], rtol=0, atol=0)


def test_lm_head_slicing(tiny_ckpt):
    model_dir, ref, cfg = tiny_ckpt
    rows = cfg.semantic_logit_rows()
    model, report = load_backbone(model_dir, device="cpu", dtype=torch.bfloat16, lm_head_rows=rows)
    assert report.lm_head_rows == len(rows) == cfg.codebook_size + 1
    assert model.lm_head.weight.shape == (len(rows), cfg.fusion_text_config().hidden_size)
    assert torch.equal(model.lm_head.weight, ref.lm_head.weight[torch.tensor(rows)])
    assert model.lm_head_rows.tolist() == rows


def test_strict_load_rejects_missing_scale(tiny_ckpt, tmp_path):
    from safetensors import safe_open

    model_dir, _, _ = tiny_ckpt
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_bytes((model_dir / "config.json").read_bytes())
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), "pt", "cpu") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys() if not k.endswith("experts.1.gate_proj.weight_scale_inv")}
        save_file(tensors, str(bad_dir / shard.name))
    with pytest.raises(LoadError):
        load_backbone(bad_dir, device="cpu", dtype=torch.bfloat16)
    model, report = load_backbone(bad_dir, device="cpu", dtype=torch.bfloat16, strict=False)
    assert report.unexpected and report.missing


def test_dequant_false_not_supported(tiny_ckpt):
    model_dir, _, _ = tiny_ckpt
    with pytest.raises(NotImplementedError):
        load_backbone(model_dir, device="cpu", dequant_experts=False)
