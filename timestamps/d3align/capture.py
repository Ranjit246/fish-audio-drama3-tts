"""Reading the hook's ``req_<n>.npz`` files and matching them to sweep requests.

File format written by hook/drama3_align_hook.py (one file per request):

    prompt_len   int64 scalar     prompt rows T seen at prefill
    layers       int64 [L]        model layer indices (3, 7, ..., 39) that captured
    steps        int64 scalar     decode steps captured (0 for warmup / profiling junk)
    logits_L<i>  float16 [steps, n_heads, T]   raw q . K_prompt^T * scaling per layer
    counter, pid, created         bookkeeping

The capture holds ALL prompt columns. Slicing to the target-text columns
happens here, before any softmax (see ``target_logits``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_REQ_RE = re.compile(r"^req_(\d+)(?:_(\d+))?\.npz$")


@dataclass
class Capture:
    path: Path
    prompt_len: int
    steps: int
    layers: tuple[int, ...]
    counter: int
    _npz: Any = field(default=None, repr=False)

    def logits(self, layer: int) -> np.ndarray:
        """Raw ``[steps, n_heads, T]`` logits of one model layer (float16 as stored)."""
        if self._npz is None:
            self._npz = np.load(self.path, allow_pickle=False)
        return self._npz[f"logits_L{layer}"]

    def all_logits(self) -> dict[int, np.ndarray]:
        return {layer: self.logits(layer) for layer in self.layers}


def capture_counter(path: Path) -> int:
    m = _REQ_RE.match(path.name)
    return int(m.group(1)) if m else -1


def read_capture(path: str | Path) -> Capture:
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    layers = tuple(int(v) for v in np.asarray(z["layers"]).reshape(-1))
    steps = int(z["steps"]) if "steps" in z.files else 0
    counter = int(z["counter"]) if "counter" in z.files else capture_counter(path)
    return Capture(path=path, prompt_len=int(z["prompt_len"]), steps=steps, layers=layers, counter=counter, _npz=z)


def list_captures(capture_dir: str | Path) -> list[Capture]:
    """All ``req_*.npz`` in counter order (junk files included; see ``is_junk``)."""
    paths = [p for p in Path(capture_dir).glob("req_*.npz") if _REQ_RE.match(p.name)]
    paths.sort(key=lambda p: (capture_counter(p), p.name))
    return [read_capture(p) for p in paths]


def is_junk(cap: Capture, prompt_lens: Iterable[int]) -> bool:
    """Warmup / profiling forwards look like prefills but never decode, or have a
    prompt length no sweep request has."""
    return cap.steps == 0 or cap.prompt_len not in set(prompt_lens)


def read_requests(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("order", 0))
    return rows


def match_captures(captures: list[Capture], requests: list[dict[str, Any]], *, step_tolerance: int = 3):
    """Pair captures with requests by ``prompt_len``, in order.

    Requests were sent sequentially at concurrency 1, so captures with decode
    steps appear in request order; each is paired with the earliest unmatched
    request of the same prompt length. Returns ``(pairs, junk, unmatched_requests)``
    where ``pairs`` is a list of ``(capture, request, notes)``.
    """
    prompt_lens = [int(r["prompt_len"]) for r in requests]
    pairs, junk = [], []
    used = set()
    for cap in captures:
        if is_junk(cap, prompt_lens):
            junk.append(cap)
            continue
        hit = None
        for i, r in enumerate(requests):
            if i not in used and int(r["prompt_len"]) == cap.prompt_len:
                hit = i
                break
        if hit is None:
            junk.append(cap)
            continue
        used.add(hit)
        r = requests[hit]
        notes = []
        exp = r.get("expected_steps")
        if exp is not None and abs(int(exp) - cap.steps) > step_tolerance:
            notes.append(f"steps {cap.steps} vs {exp} expected from audio length")
        pairs.append((cap, r, notes))
    unmatched = [r for i, r in enumerate(requests) if i not in used]
    return pairs, junk, unmatched


def target_cols_of(request: dict[str, Any]) -> np.ndarray:
    """Target-text columns of a request row (stored as ``[start, end)``)."""
    start, end = request["target_cols"]
    return np.arange(int(start), int(end), dtype=np.int64)


def target_logits(cap: Capture, target_cols: np.ndarray) -> dict[int, np.ndarray]:
    """Per layer ``[steps, n_heads, n_text]`` raw logits restricted to the target columns.

    This is the slice-BEFORE-softmax step: softmax over the slice equals
    softmax over all columns renormalised within the slice, and it keeps
    reference-transcript / system-prompt columns out of the distribution."""
    cols = np.asarray(target_cols, dtype=np.int64)
    out = {}
    for layer in cap.layers:
        M = cap.logits(layer)
        if cols.size and cols.max() >= M.shape[-1]:
            raise ValueError(f"target column {cols.max()} outside prompt width {M.shape[-1]} ({cap.path.name})")
        out[layer] = M[:, :, cols]
    return out


def selected_rows(cap: Capture, heads: list[tuple[int, int]]) -> np.ndarray:
    """``[steps, n_sel, T]`` raw logits of the given (layer, head) pairs, in order."""
    rows = []
    for layer, head in heads:
        if layer not in cap.layers:
            raise KeyError(f"layer {layer} not captured in {cap.path.name} (has {cap.layers})")
        rows.append(cap.logits(layer)[:, head, :])
    return np.stack(rows, axis=1)


__all__ = [
    "Capture",
    "capture_counter",
    "is_junk",
    "list_captures",
    "match_captures",
    "read_capture",
    "read_requests",
    "selected_rows",
    "target_cols_of",
    "target_logits",
]
