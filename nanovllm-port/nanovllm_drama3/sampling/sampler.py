# SPDX-License-Identifier: Apache-2.0
# Ported from vLLM-Omni (Apache-2.0), vllm_omni/model_executor/models/fish_omni/
# codebook_tower.py (sample_rows / sample_top_k_top_p) and fish_omni_model.py
# (FishOmniForConditionalGeneration.sample) at commit 9b300b983; candidate set
# per PR #9044. See nanovllm_drama3/tower/_reference/.
"""Drama 3 frame sampling: main token (code 0) with repetition-aware re-draw,
and top-k / top-p draws for the residual codebook heads.

Eager, CPU-friendly, B=1 oriented; every draw can take a seeded
``torch.Generator`` so a run is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

# Temperatures at or below this are greedy (vLLM marks greedy rows with -1).
GREEDY_TEMPERATURE = 1e-5

# Model-card defaults for the main token and for the codebook heads.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 30
DEFAULT_TOP_P = 0.9

# Repetition-aware re-draw (RAS): when the first semantic draw repeats one of
# the request's last RAS_WINDOW main tokens, the frame takes a second draw with
# these fixed parameters instead. Not request parameters.
RAS_WINDOW = 16
RAS_TEMPERATURE = 1.0
RAS_TOP_K = 30
RAS_TOP_P = 0.95


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P

    @property
    def greedy(self) -> bool:
        return self.temperature <= GREEDY_TEMPERATURE

    def as_tuple(self) -> tuple[float, int, float]:
        return (self.temperature, self.top_k, self.top_p)


HEAD_SAMPLING = SamplingParams(1.0, 30, 0.9)
RAS_SAMPLING = SamplingParams(RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P)


def sample_rows(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a column index per row, with per-row temperature, top-k and top-p.

    Exact port of the reference ``sample_rows``: a candidate survives when its
    rank is below top_k and the cumulative probability mass (over the whole
    temperature-scaled row, in descending order) up to and including it stays
    within top_p; the most likely candidate is never filtered out. Greedy rows
    (temperature <= GREEDY_TEMPERATURE) take the argmax. The draw uses the
    exponential-noise trick ``argmax(p / E)`` so one ``generator`` seeds it.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [rows, width], got {tuple(logits.shape)}")
    greedy = temperature <= GREEDY_TEMPERATURE
    scaled = logits.float() / temperature.float().clamp(min=GREEDY_TEMPERATURE).unsqueeze(-1)
    sorted_logits, order = torch.sort(scaled, dim=-1, descending=True)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    ranks = torch.arange(sorted_logits.shape[-1], device=logits.device).unsqueeze(0)
    drop = (cumulative > top_p.float().unsqueeze(-1)) | (ranks >= top_k.unsqueeze(-1))
    drop[:, 0] = False
    probs = torch.softmax(sorted_logits.masked_fill(drop, float("-inf")), dim=-1)
    noise = torch.empty_like(probs).exponential_(1.0, generator=generator)
    choice = torch.argmax(probs / noise, dim=-1, keepdim=True)
    choice = torch.where(greedy.unsqueeze(-1), torch.zeros_like(choice), choice)
    return torch.gather(order, -1, choice).squeeze(-1)


def sample_top_k_top_p(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """``sample_rows`` with the same scalar parameters for every row.
    ``top_k <= 0`` (or larger than the row) disables the top-k filter."""
    rows, width = logits.shape
    k = top_k if 0 < top_k <= width else width
    device = logits.device
    return sample_rows(
        logits,
        torch.full((rows,), float(temperature), device=device),
        torch.full((rows,), k, device=device, dtype=torch.long),
        torch.full((rows,), float(top_p), device=device),
        generator,
    )


def sample_codebook(
    head_logits: torch.Tensor,
    params: SamplingParams = HEAD_SAMPLING,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One residual-codebook head: ``head_logits`` [rows, codebook_size] -> codes [rows]."""
    return sample_top_k_top_p(head_logits, params.temperature, params.top_k, params.top_p, generator)


def main_candidate_ids(
    eos_token_id: int,
    semantic_begin_id: int,
    semantic_end_id: int,
    *,
    exclude_semantic_zero: bool = True,
) -> list[int]:
    """Token ids the main token is drawn from: end of speech plus the semantic band.

    With ``exclude_semantic_zero`` (vLLM-Omni PR #9044) semantic code 0 is left
    out, which equals a -inf logit bias on it: it is not a speech code, but the
    checkpoint still samples it occasionally. ``False`` reproduces the pre-#9044
    behaviour (commit 9b300b983).
    """
    first = semantic_begin_id + 1 if exclude_semantic_zero else semantic_begin_id
    return sorted({int(eos_token_id), *range(first, semantic_end_id + 1)})


@dataclass(frozen=True)
class MainTokenDraw:
    """Result of one main-token sample (vocabulary ids)."""

    token: int
    first: int
    redraw: int
    repeated: bool

    @property
    def used_redraw(self) -> bool:
        return self.repeated


class MainTokenSampler:
    """Main-token (code 0) sampling with repetition-aware re-draw (RAS).

    The row draws among end-of-speech and the semantic candidates with the
    request's temperature / top-k / top-p. When a *semantic* first draw repeats
    one of the last ``RAS_WINDOW`` generated main tokens, the row takes the
    second (RAS) draw instead; that draw may end the speech. Greedy rows take
    the argmax for the first draw; the re-draw still samples. Both draws always
    happen (in that order) so a seeded generator reproduces the reference's
    random stream.
    """

    def __init__(
        self,
        *,
        eos_token_id: int,
        semantic_begin_id: int,
        semantic_end_id: int,
        exclude_semantic_zero: bool = True,
        ras_window: int = RAS_WINDOW,
        ras_params: SamplingParams = RAS_SAMPLING,
        device: torch.device | str = "cpu",
    ) -> None:
        self.eos_token_id = int(eos_token_id)
        self.semantic_begin_id = int(semantic_begin_id)
        self.semantic_end_id = int(semantic_end_id)
        self.codebook_size = self.semantic_end_id - self.semantic_begin_id + 1
        self.ras_window = int(ras_window)
        self.ras_params = ras_params
        ids = main_candidate_ids(
            eos_token_id, semantic_begin_id, semantic_end_id, exclude_semantic_zero=exclude_semantic_zero
        )
        self.candidates = torch.tensor(ids, dtype=torch.long, device=device)

    def is_semantic(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids >= self.semantic_begin_id) & (token_ids <= self.semantic_end_id)

    def code0(self, token_id: int) -> int:
        """Semantic code of a main token (clamped; meaningless for end of speech)."""
        return min(max(token_id - self.semantic_begin_id, 0), self.codebook_size - 1)

    def candidate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Restrict full-vocabulary logits [rows, vocab] to the candidate columns."""
        return logits.index_select(-1, self.candidates.to(logits.device)).float()

    def masked_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits with every non-candidate set to -inf
        (what the reference's ``compute_logits`` returns)."""
        allowed = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
        allowed[self.candidates.to(logits.device)] = True
        return logits.masked_fill(~allowed, float("-inf"))

    def sample(
        self,
        logits: torch.Tensor,
        history: Sequence[int],
        params: SamplingParams = SamplingParams(),
        generator: torch.Generator | None = None,
    ) -> MainTokenDraw:
        """Draw the next main token from ``logits`` ([vocab] or [1, vocab]) given the
        request's generated main tokens so far (``history``, vocabulary ids)."""
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.shape[0] != 1:
            raise ValueError("MainTokenSampler.sample handles one row; use sample_rows for batches")
        device = logits.device
        cand = self.candidates.to(device)
        cand_logits = self.candidate_logits(logits)
        width = int(cand.numel())
        temperature = torch.full((1,), params.temperature if not params.greedy else -1.0, device=device)
        k = params.top_k if 0 < params.top_k <= width else width
        top_k = torch.full((1,), k, device=device, dtype=torch.long)
        top_p = torch.full((1,), params.top_p, device=device)
        first = cand[sample_rows(cand_logits, temperature, top_k, top_p, generator)]
        ras = self.ras_params
        redraw = cand[
            sample_rows(
                cand_logits,
                torch.full((1,), ras.temperature, device=device),
                torch.full((1,), ras.top_k, device=device, dtype=torch.long),
                torch.full((1,), ras.top_p, device=device),
                generator,
            )
        ]
        recent = list(history[-self.ras_window :]) if self.ras_window > 0 else []
        first_id = int(first.item())
        redraw_id = int(redraw.item())
        repeated = bool(first_id in recent) and bool(self.is_semantic(first).item())
        return MainTokenDraw(
            token=redraw_id if repeated else first_id, first=first_id, redraw=redraw_id, repeated=repeated
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
