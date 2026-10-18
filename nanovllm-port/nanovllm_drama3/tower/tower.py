# SPDX-License-Identifier: Apache-2.0
# Ported from vLLM-Omni (Apache-2.0), vllm_omni/model_executor/models/fish_omni/
# codebook_tower.py at commit 9b300b983 (eager pre-#8933 version); see
# nanovllm_drama3/tower/_reference/ for the original and VENDORED-wp2.md.
"""Drama 3 codebook tower as a plain eager PyTorch module.

The tower predicts codes 1..Q-1 of each frame from the backbone's final-norm
hidden state and the frame's main token (code 0). Every one of its
4 + (Q-1) = 13 layers attends causally over the *whole sequence* (prompt rows
and generated frames alike), so each layer keeps its own key/value history.

Sequence position ``p`` of the tower reads

    hidden[p]       backbone final-norm hidden state at row p
    frame[p]        the frame *at* row p (its codes 2..Q-1 feed shared layer 0)
    main[p + 1]     the main token of row p + 1 (sampled by the backbone from
                    hidden[p]; its semantic code feeds every layer)

and predicts codes 1..Q-1 of frame ``p + 1``. Residual layer r (0-based) takes
code r of frame p+1 as input and head r predicts code r+1, so code k conditions
on codes < k.

Module names follow the checkpoint (``audio_decoder*``) so weights load by name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm_drama3.sampling import HEAD_SAMPLING, SamplingParams, sample_codebook

from .config import TowerConfig

TOWER_PARAM_PREFIXES = ("audio_decoder.", "audio_decoder_mtp_shared.", "audio_decoder_mtp.")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class GemmaRMSNorm(nn.Module):
    """RMSNorm with a zero-centered gain (scale = 1 + weight); with ``residual``
    it normalizes ``x + residual`` and also returns that sum."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is not None:
            x = x + residual
            residual = x
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        out = (h * (1.0 + self.weight.float())).to(x.dtype)
        return out if residual is None else (out, residual)


def interleaved_rope(states: torch.Tensor, positions: torch.Tensor, rotary_dim: int, base: float) -> torch.Tensor:
    """GPT-J (interleaved) rope on the first ``rotary_dim`` channels of [rows, heads, head_dim]."""
    inv_freq = 1.0 / base ** (torch.arange(0, rotary_dim, 2, device=states.device, dtype=torch.float32) / rotary_dim)
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq  # [rows, rotary_dim/2]
    cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)  # [rows, 1, rotary_dim/2]
    rotary = states[..., :rotary_dim].float()
    even, odd = rotary[..., 0::2], rotary[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    return torch.cat((rotated.to(states.dtype), states[..., rotary_dim:]), dim=-1)


@dataclass
class LayerKV:
    """Key/value history of one tower layer: [len, kv_heads, head_dim] each."""

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    def __len__(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[0])

    def append(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None:
            self.keys, self.values = key, value
        else:
            self.keys = torch.cat((self.keys, key), dim=0)
            self.values = torch.cat((self.values, value), dim=0)
        return self.keys, self.values

    def truncate(self, length: int) -> None:
        if self.keys is not None and length < len(self):
            self.keys = self.keys[:length]
            self.values = self.values[:length]


@dataclass
class TowerKVCache:
    """One request's tower state: per-layer KV histories plus the frame that
    the next step reads as input (the last prompt frame, then the last
    generated frame)."""

    layers: list[LayerKV]
    next_position: int = 0
    input_frame: torch.Tensor | None = None  # [1 + Q] row [token_id, code0..codeQ-1]
    generated: list[torch.Tensor] = field(default_factory=list)  # frames [1 + Q] of every step

    @classmethod
    def empty(cls, num_layers: int) -> "TowerKVCache":
        return cls(layers=[LayerKV() for _ in range(num_layers)])

    def __len__(self) -> int:
        return len(self.layers[0]) if self.layers else 0


class TowerAttention(nn.Module):
    def __init__(self, config: TowerConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv = config.n_local_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.rope_base = config.rope_base
        q_width = self.n_head * self.head_dim
        kv_width = self.n_kv * self.head_dim
        # Checkpoint rows: [query | output gate | key | value].
        self.wqkv = nn.Linear(config.dim, 2 * q_width + 2 * kv_width, bias=False)
        self.wo = nn.Linear(q_width, config.dim, bias=False)
        self.q_norm = GemmaRMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, config.norm_eps)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, kv: LayerKV) -> torch.Tensor:
        """Append this call's keys/values to ``kv`` and attend causally over the
        whole history. x [rows, dim] -> [rows, dim]. ``positions`` must continue
        the cache (positions[0] == len(kv) before the call)."""
        rows = x.shape[0]
        q_width = self.n_head * self.head_dim
        kv_width = self.n_kv * self.head_dim
        query, gate, key, value = self.wqkv(x).split([q_width, q_width, kv_width, kv_width], dim=-1)
        query = self.q_norm(query.reshape(-1, self.head_dim)).view(rows, self.n_head, self.head_dim)
        key = self.k_norm(key.reshape(-1, self.head_dim)).view(rows, self.n_kv, self.head_dim)
        query = interleaved_rope(query, positions, self.rotary_dim, self.rope_base)
        key = interleaved_rope(key, positions, self.rotary_dim, self.rope_base)
        value = value.reshape(rows, self.n_kv, self.head_dim)
        keys, values = kv.append(key, value)  # [used, n_kv, head_dim]
        used = keys.shape[0]
        # Grouped-query attention: repeat each KV head for its query group.
        group = self.n_head // self.n_kv
        k_h = keys.transpose(0, 1).repeat_interleave(group, dim=0)  # [n_head, used, hd]
        v_h = values.transpose(0, 1).repeat_interleave(group, dim=0)
        q_h = query.transpose(0, 1)  # [n_head, rows, hd]
        mask = torch.arange(used, device=x.device).unsqueeze(0) <= positions.unsqueeze(1)  # [rows, used]
        attended = F.scaled_dot_product_attention(
            q_h.unsqueeze(0), k_h.unsqueeze(0), v_h.unsqueeze(0), attn_mask=mask[None, None]
        )[0]  # [n_head, rows, hd]
        attended = attended.transpose(0, 1).reshape(rows, q_width)
        return self.wo(attended * torch.sigmoid(gate))


class TowerFeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TowerBlock(nn.Module):
    def __init__(self, config: TowerConfig) -> None:
        super().__init__()
        self.attention = TowerAttention(config)
        self.feed_forward = TowerFeedForward(config.dim, config.intermediate_size)
        self.attention_norm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = GemmaRMSNorm(config.dim, config.norm_eps)


class TowerLayer(nn.Module):
    """Fuse a code embedding with the incoming hidden state (eh_proj), then apply
    one attention + feed-forward block. The embedding table is named
    ``semantic_embedding`` in shared layers and ``codebook_embedding`` in
    residual layers (checkpoint names)."""

    def __init__(self, config: TowerConfig, embedding_name: str) -> None:
        super().__init__()
        self.embedding_name = embedding_name
        self.add_module(embedding_name, nn.Embedding(config.vocab_size, config.dim))
        self.enorm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.hnorm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.eh_proj = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.transformer_layer = TowerBlock(config)
        self.final_layernorm = GemmaRMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        code_ids: torch.Tensor,
        code_is_valid: torch.Tensor,
        extra_input: torch.Tensor | None,
        positions: torch.Tensor,
        kv: LayerKV,
    ) -> torch.Tensor:
        table: nn.Embedding = getattr(self, self.embedding_name)
        embedded = torch.where(
            code_is_valid.unsqueeze(-1),
            table(code_ids.clamp(0, table.num_embeddings - 1)),
            0.0,
        )
        if extra_input is not None:
            embedded = embedded + extra_input.to(embedded.dtype)
        mixed = self.eh_proj(torch.cat((self.enorm(embedded), self.hnorm(hidden)), dim=-1))
        block = self.transformer_layer
        attended = block.attention(block.attention_norm(mixed), positions, kv)
        ffn_input, residual = block.ffn_norm(attended, mixed)
        out, _ = self.final_layernorm(block.feed_forward(ffn_input), residual)
        return out


class _LayerList(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _ResidualTower(nn.Module):
    def __init__(self, config: TowerConfig) -> None:
        super().__init__()
        count = config.num_residual_layers
        self.layers = nn.ModuleList(TowerLayer(config, "codebook_embedding") for _ in range(count))
        self.heads = nn.ModuleList(nn.Linear(config.dim, config.vocab_size, bias=False) for _ in range(count))


class _FrameEmbeddings(nn.Module):
    def __init__(self, config: TowerConfig) -> None:
        super().__init__()
        self.codebook_embeddings = nn.Embedding(config.vocab_size * config.num_codebooks, config.dim)


# ---------------------------------------------------------------------------
# The tower
# ---------------------------------------------------------------------------


@dataclass
class TowerRunOutput:
    """Codes of the predicted frames and the last layer's hidden states."""

    codes: torch.Tensor  # [rows, Q] codes 0..Q-1 of the *next* frame per row
    hidden: torch.Tensor  # [rows, dim] output of the last residual layer
    head_logits: list[torch.Tensor] | None = None  # (Q-1) x [rows, codebook_size], teacher-forced mode only


class CodebookTower(nn.Module):
    """Frame embeddings, 4 shared layers, and Q-1 residual layers with heads.

    Incremental API (one request, B = 1):

        cache = tower.new_cache()
        tower.prefill(hidden[T, dim], frames[T, 1 + Q], cache)   # prompt rows 0..T-1
        frame = tower.step(hidden_last[dim], main_token, cache)   # -> [1 + Q]
        ...

    ``frames`` rows are ``[token_id, code0, ..., codeQ-1]`` as built by
    ``nanovllm_drama3.prompt``. ``prefill`` teacher-forces rows 0..T-2 (row p
    predicts frame p+1) and keeps frame T-1 as the next step's input. ``step``
    takes the backbone hidden state of the row the main token was sampled from
    (the last prompt row for the first step, then the last generated row).
    """

    def __init__(self, config: TowerConfig, *, semantic_begin_id: int) -> None:
        super().__init__()
        self.config = config
        self.semantic_begin_id = int(semantic_begin_id)
        self.codebook_size = config.vocab_size
        self.num_codebooks = config.num_codebooks
        self.backbone_codebooks = config.vq_backbone_codebooks
        self.audio_decoder = _FrameEmbeddings(config)
        self.audio_decoder_mtp_shared = _LayerList(
            [TowerLayer(config, "semantic_embedding") for _ in range(config.shared_enrichment_layers)]
        )
        self.audio_decoder_mtp = _ResidualTower(config)
        self.register_buffer("row_offsets", torch.arange(config.num_codebooks) * config.vocab_size, persistent=False)

    # ---- helpers -----------------------------------------------------------

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    @property
    def dtype(self) -> torch.dtype:
        return self.audio_decoder.codebook_embeddings.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.audio_decoder.codebook_embeddings.weight.device

    def new_cache(self) -> TowerKVCache:
        return TowerKVCache.empty(self.num_layers)

    def is_semantic(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids >= self.semantic_begin_id) & (token_ids < self.semantic_begin_id + self.codebook_size)

    def code_embedding_sum(self, codes: torch.Tensor, first: int, end: int) -> torch.Tensor:
        """Sum of codebook embeddings of codes[:, first:end] (codebook-offset rows)."""
        rows = codes[:, first:end].clamp(0, self.codebook_size - 1) + self.row_offsets[first:end]
        return F.embedding_bag(rows, self.audio_decoder.codebook_embeddings.weight, mode="sum")

    def backbone_embedding(
        self, token_embeds: torch.Tensor, codes: torch.Tensor, is_semantic: torch.Tensor
    ) -> torch.Tensor:
        """Backbone input for frame rows: for semantic rows, the token embedding plus
        the embeddings of the first ``vq_backbone_codebooks`` residual codes,
        rescaled by 1/sqrt(1 + vq_backbone_codebooks); text rows pass through.
        token_embeds [rows, dim], codes [rows, Q], is_semantic [rows]."""
        extra = self.code_embedding_sum(codes, 1, 1 + self.backbone_codebooks).to(token_embeds.dtype)
        mixed = (token_embeds + extra) * (1.0 / math.sqrt(1.0 + self.backbone_codebooks))
        return torch.where(is_semantic.unsqueeze(-1), mixed, token_embeds)

    def backbone_input_embedding(self, token_embeds: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """``backbone_embedding`` for frame rows ``[token_id, code0..codeQ-1]``."""
        return self.backbone_embedding(token_embeds, frames[:, 1:], self.is_semantic(frames[:, 0]))

    def _shared_input(self, input_codes: torch.Tensor, input_is_semantic: torch.Tensor) -> torch.Tensor:
        """Extra input of shared layer 0: embeddings of the input frame's codes that
        the backbone did not see (codes 1 + vq_backbone_codebooks .. Q-1)."""
        extra = self.code_embedding_sum(input_codes, 1 + self.backbone_codebooks, self.num_codebooks)
        return torch.where(input_is_semantic.unsqueeze(-1), extra, 0.0)

    # ---- core --------------------------------------------------------------

    def run(
        self,
        hidden: torch.Tensor,
        input_codes: torch.Tensor,
        input_is_semantic: torch.Tensor,
        next_main_tokens: torch.Tensor,
        next_codes: torch.Tensor | None,
        cache: TowerKVCache,
        *,
        sampling: SamplingParams = HEAD_SAMPLING,
        generator: torch.Generator | None = None,
        return_head_logits: bool = False,
    ) -> TowerRunOutput:
        """Run all tower layers over ``rows`` consecutive sequence positions,
        appending to ``cache``.

        hidden [rows, dim]: backbone hidden state at each row.
        input_codes [rows, Q], input_is_semantic [rows]: the frame at each row.
        next_main_tokens [rows]: main token (vocab id) of each row's next frame.
        next_codes [rows, Q] or None: with it (prefill) codes are teacher forced;
        without it (decode) codes 1..Q-1 are sampled from the heads.
        Returns the next frames' codes [rows, Q].
        """
        rows = hidden.shape[0]
        hidden = hidden.to(self.dtype)
        positions = torch.arange(cache.next_position, cache.next_position + rows, device=hidden.device)
        label_is_semantic = self.is_semantic(next_main_tokens)
        code0 = (next_main_tokens - self.semantic_begin_id).clamp(0, self.codebook_size - 1)
        shared_input = self._shared_input(input_codes, input_is_semantic)
        layer_index = 0
        for i, layer in enumerate(self.audio_decoder_mtp_shared.layers):
            hidden = layer(
                hidden, code0, label_is_semantic, shared_input if i == 0 else None, positions, cache.layers[layer_index]
            )
            layer_index += 1
        codes = [code0]
        head_logits: list[torch.Tensor] = []
        for r, (layer, head) in enumerate(zip(self.audio_decoder_mtp.layers, self.audio_decoder_mtp.heads)):
            current = codes[-1] if next_codes is None else next_codes[:, r]
            hidden = layer(hidden, current, label_is_semantic, None, positions, cache.layers[layer_index])
            layer_index += 1
            if next_codes is None:
                codes.append(sample_codebook(head(hidden).float(), sampling, generator))
            elif return_head_logits:
                head_logits.append(head(hidden).float())
        cache.next_position += rows
        if next_codes is not None:
            return TowerRunOutput(codes=next_codes, hidden=hidden, head_logits=head_logits or None)
        return TowerRunOutput(codes=torch.stack(codes, dim=-1), hidden=hidden)

    # ---- incremental API ---------------------------------------------------

    @torch.no_grad()
    def prefill(
        self,
        hidden: torch.Tensor,
        frames: torch.Tensor,
        cache: TowerKVCache,
        *,
        return_head_logits: bool = False,
    ) -> TowerRunOutput | None:
        """Teacher-force the prompt: ``hidden`` [T, dim] and ``frames`` [T, 1 + Q].

        Rows 0..T-2 go through the tower (row p's label is frame p+1); frame T-1
        becomes the next ``step``'s input frame. Returns the run output for the
        T-1 processed rows (None when T == 1).
        """
        if hidden.ndim != 2 or frames.ndim != 2 or hidden.shape[0] != frames.shape[0]:
            raise ValueError(f"hidden {tuple(hidden.shape)} and frames {tuple(frames.shape)} must both be [T, ...]")
        if frames.shape[1] != 1 + self.num_codebooks:
            raise ValueError(f"frames must be [T, {1 + self.num_codebooks}], got {tuple(frames.shape)}")
        frames = frames.to(device=hidden.device, dtype=torch.long)
        rows = int(frames.shape[0])
        cache.input_frame = frames[-1].clone()
        if rows == 1:
            return None
        inputs, targets = frames[:-1], frames[1:]
        return self.run(
            hidden[: rows - 1],
            input_codes=inputs[:, 1:],
            input_is_semantic=self.is_semantic(inputs[:, 0]),
            next_main_tokens=targets[:, 0],
            next_codes=targets[:, 1:],
            cache=cache,
            return_head_logits=return_head_logits,
        )

    @torch.no_grad()
    def step(
        self,
        hidden: torch.Tensor,
        main_token: int | torch.Tensor,
        cache: TowerKVCache,
        *,
        sampling: SamplingParams = HEAD_SAMPLING,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample codes 1..Q-1 of the next frame.

        ``hidden`` [dim] or [1, dim]: the backbone hidden state of the row whose
        logits produced ``main_token`` (a vocabulary id). Returns the frame row
        ``[main_token, code0, ..., codeQ-1]`` (long, [1 + Q]); codes are zero
        when the main token is not semantic (end of speech). The returned frame
        becomes the input of the next step.
        """
        if cache.input_frame is None:
            raise RuntimeError("tower cache has no input frame: call prefill() first")
        hidden = hidden.reshape(1, -1)
        main = torch.as_tensor(main_token, dtype=torch.long, device=hidden.device).reshape(1)
        input_frame = cache.input_frame.to(hidden.device).unsqueeze(0)
        out = self.run(
            hidden,
            input_codes=input_frame[:, 1:],
            input_is_semantic=self.is_semantic(input_frame[:, 0]),
            next_main_tokens=main,
            next_codes=None,
            cache=cache,
            sampling=sampling,
            generator=generator,
        )
        codes = out.codes[0] * self.is_semantic(main)[0].to(out.codes.dtype)
        frame = torch.cat((main, codes))
        cache.input_frame = frame
        cache.generated.append(frame)
        return frame

    # ---- weights -----------------------------------------------------------

    def load_state(self, weights: dict[str, torch.Tensor], *, strict: bool = True) -> set[str]:
        """Copy checkpoint tensors by name (``audio_decoder*`` keys). Returns the
        names loaded; with ``strict`` every tower parameter must be present."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        with torch.no_grad():
            for name, tensor in weights.items():
                if not name.startswith(TOWER_PARAM_PREFIXES):
                    continue
                param = params.get(name)
                if param is None:
                    raise ValueError(f"unexpected tower weight {name}")
                if tuple(param.shape) != tuple(tensor.shape):
                    raise ValueError(f"{name}: shape {tuple(tensor.shape)} != {tuple(param.shape)}")
                param.copy_(tensor.to(param.dtype))
                loaded.add(name)
        missing = set(params) - loaded
        if strict and missing:
            raise ValueError(f"{len(missing)} tower weights missing, e.g. {sorted(missing)[:4]}")
        return loaded


__all__ = [
    "TOWER_PARAM_PREFIXES",
    "CodebookTower",
    "GemmaRMSNorm",
    "LayerKV",
    "TowerKVCache",
    "TowerLayer",
    "TowerRunOutput",
    "interleaved_rope",
]
