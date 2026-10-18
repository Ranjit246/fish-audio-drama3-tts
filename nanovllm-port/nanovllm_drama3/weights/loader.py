"""Load Drama 3 backbone weights into the fusion `Qwen3_5MoeForCausalLM`.

    model, report = load_backbone("/path/to/drama-3", device="cuda")

* Iterates the safetensors shards (never materialising more than one tensor
  plus pending FP8 expert pairs at a time).
* Dequantizes expert `gate_proj`/`up_proj`/`down_proj` from block-FP8 to the
  model dtype (`dequant_experts=True`; the fusion model has no FP8 kernel, so
  `False` is rejected).
* Renames via `param_map.classify` and hands tensors to the fusion
  `weight_loader`s (packed gate/up -> `gate_up_proj` shards 0/1).
* Skips `tower.*` tensors (codebook tower, package 2) and reports them.
* Optionally replaces `lm_head` by a row slice (`lm_head_rows`, e.g.
  `Drama3Config.semantic_logit_rows()`), so only 4097 x 2048 of the 252288 x 2048
  head stays resident; logits row i then corresponds to `lm_head_rows[i]`.

The only real-weight run happens on the GPU VM (package 4). CPU tests use a tiny
randomly initialised checkpoint written in the Drama 3 naming scheme.
"""

from __future__ import annotations

import glob
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import torch
from safetensors import safe_open
from torch import nn

from nanovllm_drama3.config import Drama3Config
from nanovllm_drama3.weights import param_map as pm
from nanovllm_drama3.weights.fp8 import FP8_DTYPES, dequant_block_fp8


class LoadError(RuntimeError):
    pass


@dataclass
class LoadReport:
    shards: list[str] = field(default_factory=list)
    loaded_by_class: Counter = field(default_factory=Counter)
    skipped_tower: Counter = field(default_factory=Counter)
    dequantized_experts: int = 0
    missing: list[str] = field(default_factory=list)      # model params never written
    unexpected: list[str] = field(default_factory=list)   # checkpoint tensors with no model param
    lm_head_rows: int | None = None

    @property
    def num_loaded(self) -> int:
        return sum(self.loaded_by_class.values())

    def summary(self) -> str:
        lines = [f"shards={len(self.shards)} loaded={self.num_loaded} dequantized_experts={self.dequantized_experts}"]
        for k, v in sorted(self.loaded_by_class.items()):
            lines.append(f"  {k:36s} {v}")
        if self.skipped_tower:
            lines.append("skipped tower tensors: " + ", ".join(f"{k}={v}" for k, v in sorted(self.skipped_tower.items())))
        if self.lm_head_rows is not None:
            lines.append(f"lm_head sliced to {self.lm_head_rows} rows")
        if self.missing:
            lines.append(f"MISSING ({len(self.missing)}): {self.missing[:8]}")
        if self.unexpected:
            lines.append(f"UNEXPECTED ({len(self.unexpected)}): {self.unexpected[:8]}")
        return "\n".join(lines)


def list_shards(model_dir: str | Path) -> list[str]:
    files = sorted(glob.glob(os.path.join(str(model_dir), "*.safetensors")))
    if not files:
        raise LoadError(f"no *.safetensors in {model_dir}")
    return files


def _default_loader(param: nn.Parameter, tensor: torch.Tensor) -> None:
    if param.shape != tensor.shape:
        raise LoadError(f"shape mismatch for param {tuple(param.shape)} vs tensor {tuple(tensor.shape)}")
    param.data.copy_(tensor)


def replace_lm_head_with_rows(model: nn.Module, rows: Sequence[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Swap `model.lm_head` for a ParallelLMHead over `len(rows)` vocabulary rows.
    Returns the row index tensor (kept on `model.lm_head_rows`)."""
    from nanovllm.layers.embed_head import ParallelLMHead  # vendored (import_fusion done by build_backbone)

    hidden = model.lm_head.weight.shape[1]
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            head = ParallelLMHead(len(rows), hidden)
    finally:
        torch.set_default_dtype(prev)
    rows_t = torch.as_tensor(list(rows), dtype=torch.long)
    model.lm_head = head
    model.lm_head_rows = rows_t.to(device)
    return rows_t


def load_backbone(
    model_dir: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    *,
    dequant_experts: bool = True,
    model: nn.Module | None = None,
    config: Drama3Config | None = None,
    lm_head_rows: Sequence[int] | None = None,
    strict: bool = True,
    progress: Callable[[str], None] | None = None,
) -> tuple[nn.Module, LoadReport]:
    """Build (unless `model` is given) and populate the fusion backbone from `model_dir`."""
    if not dequant_experts:
        raise NotImplementedError(
            "nano-vllm-fusion has no block-FP8 expert kernel; experts must be dequantized to the model dtype"
        )
    dev = torch.device(device)
    cfg = config or Drama3Config.from_dir(model_dir)
    block = cfg.weight_block_size

    if model is None:
        from nanovllm_drama3.backbone import build_backbone

        model = build_backbone(cfg.fusion_text_config(dtype=dtype), dev, dtype)

    report = LoadReport()
    if lm_head_rows is not None:
        rows_t = replace_lm_head_with_rows(model, lm_head_rows, dev, dtype)
        report.lm_head_rows = int(rows_t.numel())
    else:
        rows_t = None

    params = dict(model.named_parameters())
    written: set[str] = set()                 # param names fully written
    packed_seen: dict[str, set[int]] = {}     # gate_up_proj targets -> shard ids seen
    pending: dict[str, dict] = {}             # fp8 weight source -> {"w":, "s":, "entry":}

    def assign(entry: pm.ParamMapEntry, tensor: torch.Tensor) -> None:
        target = entry.target
        assert target is not None
        param = params.get(target)
        if param is None:
            report.unexpected.append(entry.source)
            return
        if entry.cls == "lm_head" and rows_t is not None:
            tensor = tensor[rows_t]
        tensor = tensor.to(dev, non_blocking=False)
        loader = getattr(param, "weight_loader", None)
        if entry.shard_id is not None:
            if loader is None:
                raise LoadError(f"{target} is packed but has no weight_loader")
            loader(param, tensor, entry.shard_id)
            seen = packed_seen.setdefault(target, set())
            seen.add(entry.shard_id)
            if seen == {0, 1}:
                written.add(target)
        else:
            (loader or _default_loader)(param, tensor)
            written.add(target)
        report.loaded_by_class[entry.cls] += 1

    def maybe_finish_fp8(src: str) -> None:
        slot = pending.get(src)
        if slot is None or slot.get("w") is None or slot.get("s") is None:
            return
        w, s, entry = slot["w"], slot["s"], slot["entry"]
        w = w.to(dev)
        s = s.to(dev)
        deq = dequant_block_fp8(w, s, block=block, out_dtype=dtype)
        report.dequantized_experts += 1
        assign(entry, deq)
        del pending[src]

    shards = list_shards(model_dir)
    for shard in shards:
        report.shards.append(os.path.basename(shard))
        if progress:
            progress(f"loading {os.path.basename(shard)}")
        with safe_open(shard, "pt", "cpu") as f:
            for name in f.keys():
                entry = pm.classify(name)
                if entry.role == pm.ROLE_SKIP:
                    report.skipped_tower[entry.cls] += 1
                    continue
                if entry.role == pm.ROLE_FP8_WEIGHT:
                    t = f.get_tensor(name)
                    if t.dtype not in FP8_DTYPES:
                        # Checkpoint variant already in bf16: load directly.
                        assign(entry, t)
                        continue
                    pending.setdefault(name, {})["w"] = t
                    pending[name]["entry"] = entry
                    maybe_finish_fp8(name)
                    continue
                if entry.role == pm.ROLE_FP8_SCALE:
                    assert entry.scale_for is not None
                    pending.setdefault(entry.scale_for, {})["s"] = f.get_tensor(name)
                    pending[entry.scale_for].setdefault("entry", pm.classify(entry.scale_for))
                    maybe_finish_fp8(entry.scale_for)
                    continue
                assign(entry, f.get_tensor(name))

    if pending:
        report.unexpected.extend(sorted(pending))
        if strict:
            raise LoadError(f"{len(pending)} FP8 expert weights missing their scale (or vice versa): {sorted(pending)[:5]}")

    report.missing = sorted(set(params) - written)
    if strict and (report.missing or report.unexpected):
        raise LoadError("incomplete load:\n" + report.summary())
    return model, report


__all__ = ["load_backbone", "LoadReport", "LoadError", "list_shards", "replace_lm_head_with_rows"]
