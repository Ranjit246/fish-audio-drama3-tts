"""Drama 3 configuration.

Loads `config.json` of `fishaudio/drama-3` (`model_type = fish_omni`) and
translates its `text_config` (Qwen3.5-MoE hybrid backbone) into the attribute
bag that nano-vllm-fusion's `Qwen3_5MoeForCausalLM` expects. transformers does
not know `fish_omni`, so the fusion `Config`/`AutoConfig` path cannot be used
directly; this module is the replacement.

A copy of the real config is bundled at `weights/drama3_meta/config.json` so
tests and tooling work offline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

META_DIR = Path(__file__).resolve().parent / "weights" / "drama3_meta"
BUNDLED_CONFIG = META_DIR / "config.json"

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"
HF_REPO_ID = "fishaudio/drama-3"


class IncompatibleConfig(ValueError):
    """The checkpoint config asks for something the fusion backbone cannot do."""


class FusionTextConfig:
    """Plain attribute bag with exactly the fields nano-vllm-fusion reads.

    Fields read by `nanovllm/models/qwen3_5_moe.py`, `qwen3_5.py`,
    `layers/gated_delta_net.py`, `engine/model_runner.py` and `config.py`:

      model_type, dtype, hidden_size, num_hidden_layers, layer_types,
      num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps,
      rope_parameters{rope_theta, partial_rotary_factor, mrope_section},
      max_position_embeddings, linear_num_value_heads, linear_num_key_heads,
      linear_key_head_dim, linear_value_head_dim, linear_conv_kernel_dim,
      num_experts, num_experts_per_tok, moe_intermediate_size,
      shared_expert_intermediate_size, intermediate_size, vocab_size,
      tie_word_embeddings, mlp_only_layers, decoder_sparse_step.

    `norm_topk_prob` is read by our patched MoE block (backbone/moe_patch.py).
    """

    _FIELDS = (
        "model_type", "dtype", "hidden_size", "num_hidden_layers", "layer_types",
        "num_attention_heads", "num_key_value_heads", "head_dim", "rms_norm_eps",
        "rope_parameters", "max_position_embeddings", "linear_num_value_heads",
        "linear_num_key_heads", "linear_key_head_dim", "linear_value_head_dim",
        "linear_conv_kernel_dim", "num_experts", "num_experts_per_tok",
        "moe_intermediate_size", "shared_expert_intermediate_size",
        "intermediate_size", "vocab_size", "tie_word_embeddings", "mlp_only_layers",
        "decoder_sparse_step", "norm_topk_prob",
    )

    def __init__(self, **kwargs: Any) -> None:
        unknown = set(kwargs) - set(self._FIELDS)
        if unknown:
            raise TypeError(f"unknown FusionTextConfig fields: {sorted(unknown)}")
        for name in self._FIELDS:
            setattr(self, name, kwargs.get(name))

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self._FIELDS}
        d["dtype"] = str(d["dtype"]).replace("torch.", "") if d["dtype"] is not None else None
        return d

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FusionTextConfig({self.to_dict()!r})"

    # Convenience views ---------------------------------------------------
    @property
    def num_full_attention_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == FULL_ATTENTION)

    @property
    def num_linear_attention_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == LINEAR_ATTENTION)

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rope_parameters["partial_rotary_factor"])


def _mrope_section(half_rotary_dim: int) -> list[int]:
    """Fusion's InterleavedMRoPE wants `sum(mrope_section) == rotary_dim // 2`.

    Drama 3 is text-only (1-D positions), so the T/H/W interleave is a no-op and
    the split is irrelevant for correctness; we only have to make the sizes
    consistent. Keep Qwen3.5's [11, 11, 10] for the real 32-pair case.
    """
    if half_rotary_dim == 32:
        return [11, 11, 10]
    third = half_rotary_dim // 3
    return [half_rotary_dim - 2 * third, third, third]


@dataclass
class Drama3Config:
    raw: dict[str, Any]
    text_config: dict[str, Any] = field(repr=False)
    audio_decoder_config: dict[str, Any] = field(repr=False)

    # ---- constructors ---------------------------------------------------
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Drama3Config":
        if raw.get("model_type") != "fish_omni":
            raise IncompatibleConfig(f"expected model_type 'fish_omni', got {raw.get('model_type')!r}")
        if "text_config" not in raw or "audio_decoder_config" not in raw:
            raise IncompatibleConfig("config.json lacks text_config / audio_decoder_config")
        return cls(raw=raw, text_config=raw["text_config"], audio_decoder_config=raw["audio_decoder_config"])

    @classmethod
    def from_file(cls, path: str | Path) -> "Drama3Config":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "Drama3Config":
        return cls.from_file(Path(model_dir) / "config.json")

    @classmethod
    def bundled(cls) -> "Drama3Config":
        """The real `fishaudio/drama-3` config shipped with this package."""
        return cls.from_file(BUNDLED_CONFIG)

    @classmethod
    def from_pretrained(cls, repo_id: str = HF_REPO_ID, **hf_kwargs: Any) -> "Drama3Config":
        """Download only config.json (never weights) from the Hub."""
        from huggingface_hub import hf_hub_download

        return cls.from_file(hf_hub_download(repo_id, "config.json", **hf_kwargs))

    # ---- token ids / codebooks -------------------------------------------
    @property
    def semantic_start_token_id(self) -> int:
        return int(self.raw["semantic_start_token_id"])

    @property
    def semantic_end_token_id(self) -> int:
        """Inclusive id of the last semantic (code-0) token."""
        return int(self.raw["semantic_end_token_id"])

    @property
    def eos_token_id(self) -> int:
        return int(self.raw["eos_token_id"])

    @property
    def pad_token_id(self) -> int:
        return int(self.raw["pad_token_id"])

    @property
    def audio_pad_token_id(self) -> int:
        return int(self.raw["audio_pad_token_id"])

    @property
    def speech_template_version(self) -> str:
        return str(self.raw.get("speech_template_version", ""))

    @property
    def num_codebooks(self) -> int:
        return int(self.audio_decoder_config["num_codebooks"])

    @property
    def codebook_size(self) -> int:
        return int(self.audio_decoder_config["vocab_size"])

    @property
    def semantic_token_range(self) -> range:
        """Backbone vocab ids that encode codebook-0 values 0..codebook_size-1."""
        r = range(self.semantic_start_token_id, self.semantic_end_token_id + 1)
        if len(r) != self.codebook_size:
            raise IncompatibleConfig(
                f"semantic token range has {len(r)} ids but codebook_size is {self.codebook_size}"
            )
        return r

    def semantic_logit_rows(self) -> list[int]:
        """Rows of lm_head the sampler needs: all code-0 tokens then EOS (last row)."""
        return list(self.semantic_token_range) + [self.eos_token_id]

    def code0_to_token_id(self, code: int) -> int:
        return self.semantic_start_token_id + code

    # ---- quantization -----------------------------------------------------
    @property
    def quant_method(self) -> str | None:
        return (self.raw.get("quantization_config") or {}).get("quant_method")

    @property
    def weight_block_size(self) -> tuple[int, int]:
        q = self.raw.get("quantization_config") or {}
        bs = q.get("weight_block_size", [128, 128])
        return int(bs[0]), int(bs[1])

    # ---- translation -------------------------------------------------------
    def fusion_text_config(self, dtype: torch.dtype = torch.bfloat16) -> FusionTextConfig:
        return fusion_text_config_from_dict(self.text_config, dtype=dtype)


def fusion_text_config_from_dict(tc: dict[str, Any], dtype: torch.dtype = torch.bfloat16) -> FusionTextConfig:
    """Translate Drama 3 `text_config` (model_type fish_omni_text) to the fusion config.

    Raises IncompatibleConfig for any architectural switch the fusion model
    hardcodes differently, so a mismatched checkpoint fails loudly instead of
    loading into the wrong graph.
    """
    def req(key: str) -> Any:
        if key not in tc:
            raise IncompatibleConfig(f"text_config lacks required key {key!r}")
        return tc[key]

    def expect(key: str, value: Any, default: Any = None) -> None:
        got = tc.get(key, default)
        if got != value:
            raise IncompatibleConfig(f"text_config.{key}={got!r}; fusion backbone requires {value!r}")

    # Switches the fusion code hardcodes.
    expect("attn_output_gate", True)              # q_proj emits [q | gate] per head
    expect("use_qk_norm", True)                   # GemmaRMSNorm on q/k heads
    expect("layernorm_zero_centered_gamma", True)  # GemmaRMSNorm (1 + w)
    expect("rotary_interleaved", False)           # half-split rotate (apply_rotary_emb)
    expect("shared_expert_gate", True)            # per-token Linear(hidden,1) gate
    expect("hidden_act", "silu", "silu")
    expect("attention_bias", False, False)

    layer_types = list(req("layer_types"))
    bad = sorted({t for t in layer_types if t not in (LINEAR_ATTENTION, FULL_ATTENTION)})
    if bad:
        raise IncompatibleConfig(f"unsupported layer_types {bad}")
    num_layers = int(req("num_hidden_layers"))
    if len(layer_types) != num_layers:
        raise IncompatibleConfig("len(layer_types) != num_hidden_layers")

    mlp_types = list(tc.get("mlp_layer_types") or ["moe"] * num_layers)
    if len(mlp_types) != num_layers:
        raise IncompatibleConfig("len(mlp_layer_types) != num_hidden_layers")
    unknown_mlp = sorted({t for t in mlp_types if t not in ("moe", "mlp")})
    if unknown_mlp:
        raise IncompatibleConfig(f"unsupported mlp_layer_types {unknown_mlp}")
    mlp_only_layers = [i for i, t in enumerate(mlp_types) if t == "mlp"]

    rope = dict(tc.get("rope_parameters") or {})
    rope_type = rope.get("rope_type", "default")
    if rope_type != "default":
        raise IncompatibleConfig(f"rope_type {rope_type!r} not supported (no scaling in fusion)")
    head_dim = int(req("head_dim"))
    partial = float(rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0)))
    rotary_dim = int(head_dim * partial)
    if rotary_dim % 2:
        raise IncompatibleConfig(f"rotary_dim {rotary_dim} must be even")
    rope_parameters = {
        "rope_theta": float(rope.get("rope_theta", tc.get("rope_theta", 10_000_000.0))),
        "partial_rotary_factor": partial,
        "mrope_section": _mrope_section(rotary_dim // 2),
    }

    num_experts = int(tc.get("num_experts", 0))
    if num_experts <= 0:
        raise IncompatibleConfig("Drama 3 backbone is MoE; num_experts must be > 0")

    shared = tc.get("shared_expert_intermediate_size")
    if shared is None:
        raise IncompatibleConfig("shared_expert_intermediate_size missing (checkpoint has shared experts)")

    return FusionTextConfig(
        model_type="qwen3_5_moe",  # makes fusion's _create_model pick Qwen3_5MoeForCausalLM
        dtype=dtype,
        hidden_size=int(req("hidden_size")),
        num_hidden_layers=num_layers,
        layer_types=layer_types,
        num_attention_heads=int(req("num_attention_heads")),
        num_key_value_heads=int(req("num_key_value_heads")),
        head_dim=head_dim,
        rms_norm_eps=float(req("rms_norm_eps")),
        rope_parameters=rope_parameters,
        max_position_embeddings=int(req("max_position_embeddings")),
        linear_num_value_heads=int(req("linear_num_value_heads")),
        linear_num_key_heads=int(req("linear_num_key_heads")),
        linear_key_head_dim=int(req("linear_key_head_dim")),
        linear_value_head_dim=int(req("linear_value_head_dim")),
        linear_conv_kernel_dim=int(req("linear_conv_kernel_dim")),
        num_experts=num_experts,
        num_experts_per_tok=int(req("num_experts_per_tok")),
        moe_intermediate_size=int(req("moe_intermediate_size")),
        shared_expert_intermediate_size=int(shared),
        intermediate_size=int(tc.get("intermediate_size", tc["moe_intermediate_size"])),
        vocab_size=int(req("vocab_size")),
        tie_word_embeddings=bool(tc.get("tie_word_embeddings", False)),
        mlp_only_layers=mlp_only_layers,
        decoder_sparse_step=None,  # every non-mlp_only layer is MoE
        norm_topk_prob=bool(tc.get("norm_topk_prob", True)),
    )


def estimate_param_bytes(cfg: FusionTextConfig, experts_dtype: torch.dtype = torch.bfloat16) -> dict[str, int]:
    """Rough parameter byte budget per component (for the GPU step)."""
    h = cfg.hidden_size
    e = cfg.moe_intermediate_size
    expert_bytes = cfg.num_experts * (3 * h * e) * torch.empty((), dtype=experts_dtype).element_size()
    n_layers = cfg.num_hidden_layers
    emb = cfg.vocab_size * h * 2
    return {
        "experts": expert_bytes * n_layers,
        "embed_tokens": emb,
        "lm_head": emb,
        "lm_head_semantic_slice": (4096 + 1) * h * 2,
    }


__all__ = [
    "Drama3Config",
    "FusionTextConfig",
    "IncompatibleConfig",
    "fusion_text_config_from_dict",
    "estimate_param_bytes",
    "BUNDLED_CONFIG",
    "META_DIR",
    "LINEAR_ATTENTION",
    "FULL_ATTENTION",
]
