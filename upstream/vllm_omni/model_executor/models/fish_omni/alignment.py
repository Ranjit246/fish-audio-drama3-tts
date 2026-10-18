# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Word alignment from Drama 3's own attention (stage 0, capture side).

A few heads of the backbone's full-attention layers read the target text in
order while the model voices it. Their attention logits over the text rows,
one row per generated frame, give word timestamps at the codec's frame rate
(44100 / 2048 = 21.53 Hz, 46.4 ms) with no aligner, no ASR and no second
model. The head set was swept offline for ``fishaudio/drama-3``
(``DEFAULT_ALIGNMENT_HEADS``, ``(layer, head)`` in model layer indices);
``hf_overrides`` ``fish_omni_alignment_heads`` replaces it.

Stage 0 replays one FULL CUDA graph per decode step, so the capture only
touches buffers allocated here, with fixed shapes and no host syncs:

* A forward pre-hook on the selected layers' ``self.attn`` (vLLM's
  ``Attention``) sees the post-norm, post-RoPE ``q`` and ``k`` of every row of
  the step, exactly as flash attention does.
* Prompt rows: the target-text rows' ``k`` (the KV heads the selected heads
  read) go into a per-slot stash ``[slots, kv, max_text, head_dim]`` through
  two static index buffers the model state fills per step (every other row
  writes into a scratch slot).
* Sample rows (one per request): ``q[head] . stash[slot, kv(head)]^T * scale``
  for the selected heads, one einsum per layer, into ``logits[slot row,
  head, :max_text]`` (fp32). Raw logits: the server slices ``[0, n_text)``
  and softmaxes, so columns past a request's text never leak into it.

The stash and the matmul inputs are fp16 (Phase 0 captured in fp16; bf16
output rounding would cost 8x the precision of the logits). GQA: 16 query
heads over 2 KV heads, ``kv = head // 8``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)

# Phase 0 head sweep of fishaudio/drama-3 (revision b2b5ee9): the six heads
# whose path mass x monotonicity x coverage over the target text ranked
# highest in both the cloned and the zero-shot condition.
DEFAULT_ALIGNMENT_HEADS: tuple[tuple[int, int], ...] = ((27, 9), (31, 4), (31, 6), (31, 0), (27, 6), (27, 0))
DEFAULT_MAX_TEXT = 512
# hf_overrides keys.
ENABLE_KEY = "fish_omni_word_timestamps"
HEADS_KEY = "fish_omni_alignment_heads"
MAX_TEXT_KEY = "fish_omni_alignment_max_text"


def parse_alignment_heads(value: Any) -> tuple[tuple[int, int], ...]:
    """``[[layer, head], ...]``, ``[{"layer": .., "head": ..}, ...]`` or ``"27:9,31:4"``."""
    if value is None:
        return DEFAULT_ALIGNMENT_HEADS
    if isinstance(value, str):
        items: Iterable[Any] = [part for part in value.replace(";", ",").split(",") if part.strip()]
    else:
        items = value
    heads: list[tuple[int, int]] = []
    for item in items:
        if isinstance(item, str):
            layer, _, head = item.strip().partition(":")
        elif isinstance(item, dict):
            layer, head = item["layer"], item["head"]
        else:
            layer, head = item
        heads.append((int(layer), int(head)))
    if not heads:
        raise ValueError(f"{HEADS_KEY} names no heads")
    if len(set(heads)) != len(heads):
        raise ValueError(f"{HEADS_KEY} repeats a head: {heads}")
    return tuple(heads)


class AlignmentHeadSet:
    """Selected heads grouped by layer, in the order the logits buffer stores them.

    ``heads`` is the configured set re-ordered by layer (stable), so each
    layer's heads occupy one contiguous block of the logits buffer. The
    server averages over heads, so the order never matters downstream.
    """

    def __init__(self, heads: Sequence[tuple[int, int]], *, num_heads: int, num_kv_heads: int) -> None:
        if num_heads % num_kv_heads:
            raise ValueError(f"{num_heads} query heads do not group over {num_kv_heads} KV heads")
        group = num_heads // num_kv_heads
        for layer, head in heads:
            if layer < 0 or not 0 <= head < num_heads:
                raise ValueError(f"alignment head (layer {layer}, head {head}) is out of range")
        self.heads = tuple(sorted(heads, key=lambda pair: pair[0]))
        self.layers = tuple(sorted({layer for layer, _ in heads}))
        # Per layer: its heads (as positions in ``self.heads``), the KV heads
        # they read (unique, stash order) and each head's position among them.
        self.layer_heads: dict[int, list[int]] = {layer: [] for layer in self.layers}
        self.layer_kv: dict[int, list[int]] = {layer: [] for layer in self.layers}
        self.layer_kv_pos: dict[int, list[int]] = {layer: [] for layer in self.layers}
        for index, (layer, head) in enumerate(self.heads):
            kv = head // group
            if kv not in self.layer_kv[layer]:
                self.layer_kv[layer].append(kv)
            self.layer_heads[layer].append(index)
            self.layer_kv_pos[layer].append(self.layer_kv[layer].index(kv))

    def __len__(self) -> int:
        return len(self.heads)


class AlignmentCapture:
    """Static buffers and the per-step maths; one instance per model."""

    def __init__(
        self,
        head_set: AlignmentHeadSet,
        *,
        max_num_tokens: int,
        max_num_seqs: int,
        max_text: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scaling: float,
        device: torch.device,
    ) -> None:
        self.head_set = head_set
        self.max_text = int(max_text)
        self.max_num_seqs = int(max_num_seqs)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.scaling = float(scaling)
        self.scratch_slot = self.max_num_seqs
        self.device = device
        long = dict(dtype=torch.int64, device=device)
        # Which stash slot and text column each token row of the step writes
        # (prompt rows of the target text); other rows write the scratch slot.
        self.row_slot = torch.full((max_num_tokens,), self.scratch_slot, **long)
        self.row_pos = torch.zeros((max_num_tokens,), **long)
        self.stash: dict[int, torch.Tensor] = {
            layer: torch.zeros(
                (self.max_num_seqs + 1, len(head_set.layer_kv[layer]), self.max_text, self.head_dim),
                dtype=torch.float16,
                device=device,
            )
            for layer in head_set.layers
        }
        self.logits = torch.zeros((self.max_num_seqs, len(head_set), self.max_text), dtype=torch.float32, device=device)
        self._index: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for layer in head_set.layers:
            heads = torch.tensor([head_set.heads[i][1] for i in head_set.layer_heads[layer]], **long)
            kv = torch.tensor(head_set.layer_kv[layer], **long)
            pick = torch.arange(len(heads), **long)
            kv_pos = torch.tensor(head_set.layer_kv_pos[layer], **long)
            self._index[layer] = (heads, kv, pick, kv_pos)
        # Set by the model's forward before the backbone runs: the sample-row
        # count (a SymInt under compile) and the sample rows' token rows / slots.
        self.num_samples: Any = 0
        self.sample_rows: torch.Tensor | None = None
        self.sample_slots: torch.Tensor | None = None

    def begin_step(self, num_samples: Any, sample_rows: torch.Tensor, sample_slots: torch.Tensor) -> None:
        self.num_samples = num_samples
        self.sample_rows = sample_rows
        self.sample_slots = sample_slots

    def capture(self, layer: int, q: torch.Tensor, k: torch.Tensor) -> None:
        """Stash the step's target-text keys and score the sample rows (one layer).

        ``q`` is ``[rows, num_heads * head_dim]`` and ``k`` ``[rows,
        num_kv_heads * head_dim]`` for every token row of the step (padding
        rows included: they write the scratch slot).
        """
        heads, kv, pick, kv_pos = self._index[layer]
        stash = self.stash[layer]
        n = k.shape[0]
        keys = k.reshape(n, self.num_kv_heads, self.head_dim)[:, kv].to(torch.float16)  # [n, n_kv, D]
        stash[self.row_slot[:n], :, self.row_pos[:n]] = keys
        s = self.num_samples
        rows = self.sample_rows[:s]
        slots = self.sample_slots[:s]
        queries = q[rows].reshape(-1, self.num_heads, self.head_dim)[:, heads].to(torch.float16)  # [S, n_h, D]
        # All (head, stashed kv) pairs, then each head's own KV head: a handful
        # of heads, so the extra pairs cost less than a second gather.
        pairs = torch.einsum("shd,sktd->shkt", queries, stash[slots])  # [S, n_h, n_kv, T]
        scores = pairs[:, pick, kv_pos].float() * self.scaling  # [S, n_h, T]
        first = self.head_set.layer_heads[layer][0]
        self.logits[:s, first : first + len(heads)] = scores

    def extract(self, rows: torch.Tensor, n_text: int) -> torch.Tensor:
        """Owned fp16 ``[len(rows), heads, n_text]`` of the given sample rows' logits.

        ``rows`` is a device int64 tensor (uploaded without a host sync by the
        model state, like its other per-step inputs).
        """
        return self.logits.index_select(0, rows)[:, :, :n_text].to(torch.float16)


def alignment_config(config: Any) -> tuple[bool, tuple[tuple[int, int], ...], int]:
    """``(enabled, heads, max_text)`` from the HF config (``hf_overrides``)."""
    enabled = bool(getattr(config, ENABLE_KEY, False))
    heads = parse_alignment_heads(getattr(config, HEADS_KEY, None))
    max_text = int(getattr(config, MAX_TEXT_KEY, DEFAULT_MAX_TEXT))
    if max_text <= 0:
        raise ValueError(f"{MAX_TEXT_KEY}={max_text}: needs a positive text-column capacity")
    return enabled, heads, max_text


def _layer_index(module: nn.Module, fallback: int) -> int:
    for name in (getattr(getattr(module, "attn", None), "layer_name", None), getattr(module, "prefix", None)):
        if isinstance(name, str) and ".layers." in name:
            try:
                return int(name.split(".layers.")[1].split(".")[0])
            except ValueError:
                pass
    return fallback


def install_alignment_hooks(
    layers: Sequence[nn.Module],
    capture: AlignmentCapture,
    *,
    attention_attr: str = "self_attn",
) -> dict[int, Callable[..., Any]]:
    """Register the capture pre-hook on ``layers[i].<attention_attr>.attn`` of the selected layers.

    Must run before torch.compile / CUDA graph capture: the hooks become part
    of the traced forward and only touch ``capture``'s static buffers.
    Returns the hooks by layer index.
    """
    hooks: dict[int, Callable[..., Any]] = {}
    for layer in capture.head_set.layers:
        if layer >= len(layers):
            raise ValueError(f"alignment head set names layer {layer}; the backbone has {len(layers)} layers")
        attention = getattr(layers[layer], attention_attr, None)
        if attention is None or not hasattr(attention, "attn"):
            raise ValueError(f"backbone layer {layer} has no full attention ({attention_attr}.attn); choose another head")
        found = _layer_index(attention, layer)
        if found != layer:
            raise ValueError(f"backbone layer {layer} reports layer index {found}")
        for name, expected in (
            ("num_heads", capture.num_heads),
            ("num_kv_heads", capture.num_kv_heads),
            ("head_dim", capture.head_dim),
        ):
            value = getattr(attention, name, expected)
            if int(value) != expected:
                raise ValueError(f"backbone layer {layer}: {name}={value}, alignment capture expects {expected}")

        def hook(module: nn.Module, args: tuple[Any, ...], _layer: int = layer) -> None:
            capture.capture(_layer, args[0], args[1])

        attention.attn.register_forward_pre_hook(hook)
        hooks[layer] = hook
    logger.info(
        "fish_omni: word-timestamp capture on heads %s (max %d text tokens, %.1f MiB of stash)",
        ", ".join(f"L{layer}H{head}" for layer, head in capture.head_set.heads),
        capture.max_text,
        sum(t.numel() * t.element_size() for t in capture.stash.values()) / 2**20,
    )
    return hooks


__all__ = [
    "DEFAULT_ALIGNMENT_HEADS",
    "DEFAULT_MAX_TEXT",
    "ENABLE_KEY",
    "HEADS_KEY",
    "MAX_TEXT_KEY",
    "AlignmentCapture",
    "AlignmentHeadSet",
    "alignment_config",
    "install_alignment_hooks",
    "parse_alignment_heads",
]
