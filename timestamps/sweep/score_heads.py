#!/usr/bin/env python3
"""Score all 160 (layer, head) pairs and pick the alignment heads.

For every matched capture and every layer: slice the raw logits to the
TARGET-TEXT columns, softmax over those columns, then per head
``path_mass``, ``monotonicity``, ``coverage`` and
``score = path_mass * monotonicity * min(2 * coverage, 1)`` (VoxCPM2's
align_sweep.py definitions; DP in numba, numpy fallback). Metrics are averaged
over the utterances of each condition (cloned / zeroshot) and written to
``sweep_<condition>.json`` (all heads). ``alignment_heads.json`` ranks heads by
the MEAN score across conditions and keeps the top k.

Also reported per head: ``target_mass``, the share of the full-prompt softmax
that lands on the target columns. The VoxCPM2 README's warning applies: a head
can look diagonal after renormalisation while putting almost all of its mass
on the reference transcript. Check it for the chosen heads (uniform would be
n_text / prompt_len; good heads were 7x uniform on VoxCPM2).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d3align import FULL_ATTENTION_LAYERS, NUM_HEADS  # noqa: E402
from d3align.capture import list_captures, match_captures, read_capture, read_requests, target_cols_of  # noqa: E402
from d3align.dp import HAVE_NUMBA  # noqa: E402
from d3align.scoring import composite, layer_metrics  # noqa: E402
from d3align.tokenizer import model_revision  # noqa: E402

METRICS = ("path_mass", "monotonicity", "coverage", "target_mass")


def score_pairs(pairs, *, verbose: bool = True):
    """-> {condition: {(layer, head): {metric: mean, "n_utts": int}}}, per-utt uniform baselines."""
    acc: dict[str, dict[tuple[int, int], dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    baselines: dict[str, list[float]] = defaultdict(list)
    for cap, req, notes in pairs:
        cond = req.get("condition", "all")
        cols = target_cols_of(req)
        if cols.size == 0 or cap.steps == 0:
            continue
        baselines[cond].append(cols.size / cap.prompt_len)
        t0 = time.perf_counter()
        for layer in cap.layers:
            mets, _ = layer_metrics(cap.logits(layer), cols)
            for h, m in enumerate(mets):
                for k in METRICS:
                    acc[cond][(layer, h)][k].append(m[k])
        if verbose:
            print(
                f"  [{req.get('order', '?')}] {cond} {req.get('utt', '')}: steps={cap.steps} n_text={cols.size} "
                f"prompt={cap.prompt_len} ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
    out = {}
    for cond, heads in acc.items():
        rows = {}
        for key, lists in heads.items():
            r = {k: float(np.mean(v)) for k, v in lists.items()}
            r["score"] = composite(r["path_mass"], r["monotonicity"], r["coverage"])
            r["n_utts"] = len(lists["path_mass"])
            rows[key] = r
        out[cond] = rows
    return out, {c: float(np.mean(v)) for c, v in baselines.items()}


def sweep_json(cond: str, rows: dict, meta: dict) -> dict:
    heads = [{"layer": l, "head": h, **r} for (l, h), r in rows.items()]
    heads.sort(key=lambda r: -r["score"])
    return {"meta": {**meta, "condition": cond}, "heads": heads}


def rank_heads(per_cond: dict, topk: int) -> list[dict]:
    conds = sorted(per_cond)
    common = set.intersection(*(set(per_cond[c]) for c in conds)) if conds else set()
    ranked = sorted(common, key=lambda k: -float(np.mean([per_cond[c][k]["score"] for c in conds])))
    out = []
    for layer, head in ranked[:topk]:
        e = {
            "layer": int(layer),
            "head": int(head),
            "score": round(float(np.mean([per_cond[c][(layer, head)]["score"] for c in conds])), 4),
            "target_mass": round(float(np.mean([per_cond[c][(layer, head)]["target_mass"] for c in conds])), 4),
        }
        for c in conds:
            r = per_cond[c][(layer, head)]
            e[c] = {k: round(r[k], 4) for k in ("score", "path_mass", "monotonicity", "coverage", "target_mass")}
        out.append(e)
    return out


def print_top(cond: str, rows: dict, baseline: float, n: int = 20) -> None:
    heads = sorted(rows.items(), key=lambda kv: -kv[1]["score"])
    print(f"\n== {cond}: top {n} of {len(rows)} heads (uniform path_mass ~ 1/n_text; uniform target_mass ~ {baseline:.3f}) ==")
    print(f"{'layer':>5} {'head':>4} {'score':>7} {'path_mass':>9} {'monotonic':>9} {'coverage':>8} {'tgt_mass':>8}")
    for (l, h), r in heads[:n]:
        print(f"{l:>5} {h:>4} {r['score']:>7.4f} {r['path_mass']:>9.4f} {r['monotonicity']:>9.3f} {r['coverage']:>8.3f} {r['target_mass']:>8.3f}")
    scores = np.array([r["score"] for r in rows.values()])
    print(f"   median score {np.median(scores):.4f}, best {scores.max():.4f} ({scores.max() / max(np.median(scores), 1e-9):.1f}x median)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture-dir", required=True, help="$DRAMA3_ALIGN_OUT of the server run")
    ap.add_argument("--requests", default=str(ROOT / "sweep" / "out" / "requests.jsonl"))
    ap.add_argument("--out-dir", default=None, help="where sweep_<cond>.json / alignment_heads.json go (default: requests dir)")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--model", default="fishaudio/drama-3")
    ap.add_argument("--revision", default=None, help="checkpoint revision recorded in alignment_heads.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    requests = read_requests(args.requests)
    captures = list_captures(args.capture_dir)
    pairs, junk, unmatched = match_captures(captures, requests)
    if not pairs:
        print("no capture matched a request; run sweep/match.py to see why", file=sys.stderr)
        return 1
    print(f"{len(pairs)} matched captures, {len(junk)} junk skipped, {len(unmatched)} requests without capture; numba={HAVE_NUMBA}")
    per_cond, baselines = score_pairs(pairs, verbose=not args.quiet)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.requests).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    revision = args.revision or model_revision(args.model) or "unknown"
    meta = {
        "model": args.model,
        "revision": revision,
        "layer_index_space": "model layer index 0..39 (full-attention layers %s)" % list(FULL_ATTENTION_LAYERS),
        "n_heads_per_layer": NUM_HEADS,
        "score": "path_mass * monotonicity * min(2*coverage, 1), target columns only, mean over utterances",
        "numba": HAVE_NUMBA,
    }
    for cond, rows in per_cond.items():
        p = out_dir / f"sweep_{cond}.json"
        p.write_text(json.dumps(sweep_json(cond, rows, {**meta, "n_utts": max(r["n_utts"] for r in rows.values()), "uniform_target_mass": baselines[cond]}), indent=1))
        print_top(cond, rows, baselines[cond])
        print(f"wrote {p}")

    heads = rank_heads(per_cond, args.topk)
    conds = sorted(per_cond)
    doc = {
        "note": f"alignment heads for {args.model}@{revision}; ranked by mean score across {len(conds)} condition(s): {', '.join(conds)}",
        **meta,
        "conditions": conds,
        "uniform_target_mass": baselines,
        "heads": heads,
    }
    hp = out_dir / "alignment_heads.json"
    hp.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"\n== top-{args.topk} by mean across {conds} -> {hp} ==")
    for e in heads:
        per = "  ".join(f"{c}={e[c]['score']:.3f}" for c in conds)
        flag = ""
        base = float(np.mean(list(baselines.values())))
        if e["target_mass"] < 2 * base:
            flag = "   <-- target_mass barely above uniform: this head mostly reads the reference/system prompt"
        print(f"   L{e['layer']:<2} H{e['head']:<2} mean={e['score']:.4f}  {per}  target_mass={e['target_mass']:.3f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
