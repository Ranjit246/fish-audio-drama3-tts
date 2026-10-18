"""Map Drama 3 safetensors parameter names to nano-vllm-fusion module paths.

Source names come from `model.safetensors.index.json` of `fishaudio/drama-3`
(62,245 tensors, 8 shards). Every name falls in one of these classes:

  backbone.*  -> loaded into `Qwen3_5MoeForCausalLM` (target path given)
  lm_head     -> `lm_head.weight`
  tower.*     -> codebook tower (`audio_decoder*`), owned by package 2; we only
                 classify and prefix the target with `tower.`

Packed weights: fusion fuses `gate_proj`+`up_proj` into `gate_up_proj` and loads
them through `MergedColumnParallelLinear.weight_loader(param, tensor, shard_id)`
with shard_id 0 (gate) / 1 (up). FP8 experts carry a companion
`weight_scale_inv` (F32, per 128x128 block) whose target is the *same* fused
parameter; the loader dequantizes before calling the weight_loader.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

META_DIR = Path(__file__).resolve().parent / "drama3_meta"
BUNDLED_INDEX = META_DIR / "model.safetensors.index.json.gz"
BUNDLED_SHAPES = META_DIR / "tensor_shapes.json.gz"
PARAM_MAP_JSON = Path(__file__).resolve().parent / "param_map.json"

BACKBONE_PREFIX = "model.language_model."

ROLE_WEIGHT = "weight"          # plain tensor -> weight_loader(param, t)
ROLE_PACKED = "packed"          # -> weight_loader(param, t, shard_id)
ROLE_FP8_WEIGHT = "fp8_weight"  # e4m3 tensor; needs its scale, then packed load
ROLE_FP8_SCALE = "fp8_scale"    # block scale_inv for an fp8 weight
ROLE_SKIP = "skip"              # tower.* (not part of the backbone module)


class UnknownParam(KeyError):
    """A checkpoint tensor name no rule understands."""


@dataclass(frozen=True)
class ParamMapEntry:
    source: str
    cls: str
    target: str | None           # fusion module parameter path (or `tower.`-prefixed)
    role: str
    shard_id: int | None = None  # for packed gate/up: 0 / 1
    scale_for: str | None = None  # for fp8_scale: source name of the weight it scales
    layer: int | None = None
    expert: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# (regex, class, role) -- regexes are anchored on the full name.
_L = r"(?P<layer>\d+)"
_E = r"(?P<expert>\d+)"
_BB = re.escape(BACKBONE_PREFIX)
_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(rf"{_BB}embed_tokens\.weight"), "backbone.embed_tokens", ROLE_WEIGHT),
    (re.compile(rf"{_BB}norm\.weight"), "backbone.final_norm", ROLE_WEIGHT),
    (re.compile(rf"{_BB}layers\.{_L}\.(input_layernorm|post_attention_layernorm)\.weight"), "backbone.layer_norm", ROLE_WEIGHT),
    (re.compile(rf"{_BB}layers\.{_L}\.self_attn\.(q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)\.weight"), "backbone.full_attn", ROLE_WEIGHT),
    (re.compile(rf"{_BB}layers\.{_L}\.linear_attn\.(A_log|dt_bias|conv1d\.weight|in_proj_qkv\.weight|in_proj_z\.weight|in_proj_a\.weight|in_proj_b\.weight|norm\.weight|out_proj\.weight)"), "backbone.linear_attn", ROLE_WEIGHT),
    (re.compile(rf"{_BB}layers\.{_L}\.mlp\.gate\.weight"), "backbone.moe_router", ROLE_WEIGHT),
    (re.compile(rf"{_BB}layers\.{_L}\.mlp\.shared_expert_gate\.weight"), "backbone.moe_shared_expert_gate", ROLE_WEIGHT),
    (re.compile(rf"{_BB}layers\.{_L}\.mlp\.shared_expert\.(gate_proj|up_proj)\.weight"), "backbone.moe_shared_expert", ROLE_PACKED),
    (re.compile(rf"{_BB}layers\.{_L}\.mlp\.shared_expert\.down_proj\.weight"), "backbone.moe_shared_expert", ROLE_WEIGHT),
    (re.compile(rf"{_BB}layers\.{_L}\.mlp\.experts\.{_E}\.(gate_proj|up_proj|down_proj)\.weight_scale_inv"), "backbone.moe_expert_scale", ROLE_FP8_SCALE),
    (re.compile(rf"{_BB}layers\.{_L}\.mlp\.experts\.{_E}\.(gate_proj|up_proj|down_proj)\.weight"), "backbone.moe_expert_fp8", ROLE_FP8_WEIGHT),
    (re.compile(r"lm_head\.weight"), "lm_head", ROLE_WEIGHT),
    (re.compile(r"audio_decoder\.codebook_embeddings\.weight"), "tower.codebook_embeddings", ROLE_SKIP),
    (re.compile(r"audio_decoder_mtp_shared\.layers\.\d+\..+"), "tower.shared", ROLE_SKIP),
    (re.compile(r"audio_decoder_mtp\.layers\.\d+\..+"), "tower.residual", ROLE_SKIP),
    (re.compile(r"audio_decoder_mtp\.heads\.\d+\.weight"), "tower.heads", ROLE_SKIP),
]

CLASSES: tuple[str, ...] = tuple(dict.fromkeys(cls for _, cls, _ in _RULES))
_PACKED = {"gate_proj": 0, "up_proj": 1}


def _backbone_target(source: str) -> tuple[str, int | None]:
    """`model.language_model.X` -> `model.X`, with gate/up fused into gate_up_proj."""
    name = "model." + source[len(BACKBONE_PREFIX):]
    for proj, shard in _PACKED.items():
        token = f".{proj}.weight"
        if name.endswith(token):
            return name[: -len(token)] + ".gate_up_proj.weight", shard
    return name, None


def classify(source: str) -> ParamMapEntry:
    for rx, cls, role in _RULES:
        m = rx.fullmatch(source)
        if not m:
            continue
        gd = m.groupdict()
        layer = int(gd["layer"]) if gd.get("layer") is not None else None
        expert = int(gd["expert"]) if gd.get("expert") is not None else None
        if role == ROLE_SKIP:
            return ParamMapEntry(source, cls, "tower." + source, role, layer=layer, expert=expert)
        if cls == "lm_head":
            return ParamMapEntry(source, cls, "lm_head.weight", role)
        if role == ROLE_FP8_SCALE:
            weight_src = source[: -len("_scale_inv")]
            target, shard = _backbone_target(weight_src)
            return ParamMapEntry(source, cls, target, role, shard_id=shard, scale_for=weight_src, layer=layer, expert=expert)
        target, shard = _backbone_target(source)
        if role == ROLE_WEIGHT and shard is not None:
            role = ROLE_PACKED
        return ParamMapEntry(source, cls, target, role, shard_id=shard, layer=layer, expert=expert)
    raise UnknownParam(source)


# --------------------------------------------------------------------------
# index helpers
# --------------------------------------------------------------------------
def load_index(path: str | Path | None = None) -> dict[str, str]:
    """`weight_map` {tensor name -> shard file} from an index json (optionally .gz)."""
    p = Path(path) if path is not None else BUNDLED_INDEX
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as fh:
        return json.load(fh)["weight_map"]


def load_shapes(path: str | Path | None = None) -> dict[str, dict]:
    """{tensor name -> {"dtype": "BF16"|"F8_E4M3"|"F32", "shape": [...]}} read from the
    safetensors headers (fetched once by HTTP range request; no weights)."""
    p = Path(path) if path is not None else BUNDLED_SHAPES
    with gzip.open(p, "rt") as fh:
        return json.load(fh)


def build_param_map(names: Iterable[str]) -> dict[str, ParamMapEntry]:
    out: dict[str, ParamMapEntry] = {}
    unknown: list[str] = []
    for n in names:
        try:
            out[n] = classify(n)
        except UnknownParam:
            unknown.append(n)
    if unknown:
        raise UnknownParam(f"{len(unknown)} unclassified tensors, e.g. {unknown[:5]}")
    return out


def _pattern(name: str) -> str:
    name = re.sub(r"\.layers\.\d+\.", ".layers.{layer}.", name)
    name = re.sub(r"\.experts\.\d+\.", ".experts.{expert}.", name)
    name = re.sub(r"\.heads\.\d+\.", ".heads.{head}.", name)
    return name


def summarize(param_map: dict[str, ParamMapEntry], shapes: dict[str, dict] | None = None) -> dict:
    """Collapsed, human-readable map: per-class counts plus one row per name pattern."""
    counts = Counter(e.cls for e in param_map.values())
    rows: dict[tuple, dict] = {}
    for e in param_map.values():
        key = (_pattern(e.source), e.cls, _pattern(e.target or ""), e.role, e.shard_id)
        row = rows.setdefault(key, {
            "source": key[0], "class": e.cls, "target": key[2], "role": e.role,
            **({"shard_id": e.shard_id} if e.shard_id is not None else {}),
            "count": 0,
        })
        row["count"] += 1
        if shapes and e.source in shapes and "dtype" not in row:
            row["dtype"] = shapes[e.source]["dtype"]
            row["shape"] = shapes[e.source]["shape"]
    return {
        "num_tensors": len(param_map),
        "classes": dict(sorted(counts.items())),
        "patterns": sorted(rows.values(), key=lambda r: (r["class"], r["source"])),
    }


def write_param_map_json(path: str | Path = PARAM_MAP_JSON, full: bool = False) -> dict:
    pm = build_param_map(load_index())
    shapes = load_shapes() if BUNDLED_SHAPES.exists() else None
    doc = summarize(pm, shapes)
    if full:
        doc["entries"] = {k: v.to_dict() for k, v in pm.items()}
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    return doc


if __name__ == "__main__":  # python -m nanovllm_drama3.weights.param_map
    d = write_param_map_json()
    print(json.dumps(d["classes"], indent=1))
