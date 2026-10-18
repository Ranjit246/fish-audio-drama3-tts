#!/usr/bin/env python3
"""Word timestamps from one captured request, with the frozen head set.

    align_decode.py --npz req_3.npz --requests requests.jsonl --heads alignment_heads.json

Finds the request the capture belongs to (by prompt_len, in order, the same way
sweep/match.py does), stacks the selected heads' raw ``[steps, T]`` rows into
``[steps, n_sel, T]``, then ``reduce_matrix`` (slice to target columns ->
softmax -> z-score per head over time -> mean -> minus min), ``dtw_path``
(numba) and ``words_from_path`` via the tokenizer's offset mapping of the
target text. Prints the spans and writes ``<npz>.words.json`` (or ``--out``).

The public functions are re-exported from ``d3align.decode`` / ``d3align.dp``:
``reduce_matrix(raw_logits[steps, nsel, T], target_cols) -> A[steps, n_text]``,
``dtw_path(A) -> path[steps]``, ``words_from_path(path, text, tokenizer_offsets,
frame_ms=46.44) -> [{word, start_ms, end_ms}]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d3align import FRAME_MS  # noqa: E402
from d3align.capture import (  # noqa: E402
    list_captures,
    match_captures,
    read_capture,
    read_requests,
    selected_rows,
    target_cols_of,
)
from d3align.decode import decode_words, reduce_matrix, target_token_offsets, words_from_path  # noqa: E402,F401
from d3align.dp import HAVE_NUMBA, dtw_path  # noqa: E402,F401
from d3align.tokenizer import load_tokenizer  # noqa: E402


def load_heads(path: str | Path, topk: int | None) -> list[tuple[int, int]]:
    d = json.loads(Path(path).read_text())
    rows = d["heads"] if isinstance(d, dict) else d
    heads = [(int(r["layer"]), int(r["head"])) for r in rows]
    return heads[:topk] if topk else heads


def find_request(npz: Path, requests_path: Path) -> dict:
    """The request record of ``npz``: matched in order over the whole capture dir."""
    cap = read_capture(npz)
    requests = read_requests(requests_path)
    pairs, _, _ = match_captures(list_captures(npz.parent), requests)
    for c, r, _notes in pairs:
        if c.path.resolve() == npz.resolve():
            return r
    # fall back to the first request with the same prompt length
    for r in requests:
        if int(r["prompt_len"]) == cap.prompt_len:
            return r
    raise SystemExit(f"no request with prompt_len={cap.prompt_len} in {requests_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--heads", required=True, help="alignment_heads.json from score_heads.py")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--tokenizer", default="fishaudio/drama-3")
    ap.add_argument("--frame-ms", type=float, default=FRAME_MS)
    ap.add_argument("--frame-offset", type=int, default=0, help="shift all spans by N frames (see d3align.decode)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    npz = Path(args.npz)
    cap = read_capture(npz)
    req = find_request(npz, Path(args.requests))
    heads = load_heads(args.heads, args.topk)
    tok = load_tokenizer(args.tokenizer)
    cols = target_cols_of(req)
    raw = selected_rows(cap, heads)  # [steps, n_sel, T]
    words, path, A = decode_words(raw, cols, req["text"], tok, frame_ms=args.frame_ms, frame_offset=args.frame_offset)

    print(f"{npz.name}: {cap.steps} steps x {len(cols)} target columns ({cap.steps * args.frame_ms / 1000:.2f}s), "
          f"heads {', '.join(f'L{l}H{h}' for l, h in heads)}, numba={HAVE_NUMBA}")
    print(f"text: {req['text']!r}  ({req['condition']}, audio {req.get('audio_s', float('nan')):.2f}s)")
    print(f"\n{'word':<16} {'start':>8} {'end':>8}   columns")
    print("-" * 50)
    for w in words:
        s = "-" if w["start_ms"] is None else f"{w['start_ms'] / 1000:8.3f}"
        e = "-" if w["end_ms"] is None else f"{w['end_ms'] / 1000:8.3f}"
        tag = "  (interpolated)" if w.get("interpolated") else ""
        print(f"{w['word']:<16} {s:>8} {e:>8}   {w['columns']}{tag}")

    out = Path(args.out) if args.out else npz.with_suffix(".words.json")
    out.write_text(
        json.dumps(
            {
                "npz": str(npz),
                "utt": req.get("utt"),
                "condition": req.get("condition"),
                "text": req["text"],
                "heads": [{"layer": l, "head": h} for l, h in heads],
                "frame_ms": args.frame_ms,
                "frame_offset": args.frame_offset,
                "steps": int(cap.steps),
                "path": [int(p) for p in path],
                "words": words,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
