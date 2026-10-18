#!/usr/bin/env python3
"""Pair the hook's req_<n>.npz captures with run_sweep.py's requests.jsonl.

Matching key: ``prompt_len`` (the prefill row count the hook saw == the row
count ``prompt_length()`` predicted), resolved in order since requests were
sequential. Captures with no decode steps or with a prompt length no request
has are vLLM warmup / profiling junk and are skipped.

Writes ``matched.jsonl`` (one line per pair: npz path + the request record +
capture steps) and prints a table including the steps-vs-audio sanity check.
The target-column slice itself is ``d3align.capture.target_logits``; this
script only verifies the columns fit the capture width.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d3align.capture import list_captures, match_captures, read_requests, target_cols_of, target_logits  # noqa: E402


def run(capture_dir: Path, requests_path: Path, out_path: Path | None, verbose: bool = True):
    captures = list_captures(capture_dir)
    requests = read_requests(requests_path)
    pairs, junk, unmatched = match_captures(captures, requests)
    rows = []
    for cap, req, notes in pairs:
        cols = target_cols_of(req)
        try:
            sliced = target_logits(cap, cols)
            n_text = next(iter(sliced.values())).shape[-1] if sliced else 0
        except ValueError as e:
            notes = notes + [str(e)]
            n_text = 0
        rec = {
            **req,
            "npz": str(cap.path),
            "steps": cap.steps,
            "layers": list(cap.layers),
            "n_text_sliced": n_text,
            "notes": notes,
        }
        rows.append(rec)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    if verbose:
        print(f"{len(captures)} capture files: {len(pairs)} matched, {len(junk)} junk (warmup/profiling/unknown prompt_len)")
        print(f"{'#':>3} {'cond':<8} {'prompt':>6} {'steps':>5} {'~audio':>6} {'ntext':>5}  utt / notes")
        for r in rows:
            exp = r.get("expected_steps")
            print(
                f"{r['order']:>3} {r['condition']:<8} {r['prompt_len']:>6} {r['steps']:>5} {str(exp):>6} "
                f"{r['n_text_sliced']:>5}  {r['utt']}" + (f"  !! {'; '.join(r['notes'])}" if r["notes"] else "")
            )
        for cap in junk:
            print(f"    junk: {cap.path.name} prompt_len={cap.prompt_len} steps={cap.steps}")
        for r in unmatched:
            print(f"    UNMATCHED request {r['order']} ({r['condition']}, prompt_len={r['prompt_len']}): {r['utt']}")
        if out_path is not None:
            print(f"wrote {out_path}")
    return rows, junk, unmatched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture-dir", required=True, help="$DRAMA3_ALIGN_OUT of the server run")
    ap.add_argument("--requests", default=str(ROOT / "sweep" / "out" / "requests.jsonl"))
    ap.add_argument("--out", default=None, help="matched.jsonl (default: next to requests.jsonl)")
    args = ap.parse_args()
    req = Path(args.requests)
    out = Path(args.out) if args.out else req.parent / "matched.jsonl"
    rows, _, unmatched = run(Path(args.capture_dir), req, out)
    return 0 if rows and not unmatched else 1


if __name__ == "__main__":
    raise SystemExit(main())
