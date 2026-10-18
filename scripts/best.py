"""Profile x concurrency matrix from runs/*/summary.json, and the best profile per
concurrency under a stall budget.

Criterion: among profiles whose audible-stall rate (>100 ms underrun) is <= BUDGET %,
pick the lowest TTFA p50; ties -> lower RTF. Also prints the single profile that is
admissible at every measured level with the lowest mean TTFA (the "one deploy" pick).

Usage: best.py [runs_dir] [--budget 1.0]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

H200 = {1: 66, 2: 70, 4: 75, 8: 82, 16: 104, 32: 135, 64: 198, 128: 255}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default=str(Path.home() / "drama3/runs"))
    ap.add_argument("--budget", type=float, default=1.0, help="max audible-stall %% to be admissible")
    args = ap.parse_args()

    cells: dict[str, dict[int, dict]] = defaultdict(dict)
    for s in Path(args.runs).glob("*/summary.json"):
        m = re.fullmatch(r"(.+)_c(\d+)", s.parent.name)
        if not m or m.group(1) in ("smoke",) or m.group(1).endswith("_audio"):
            continue
        d = json.loads(s.read_text())
        # final_* runs are the same profile on all 1088 lines; prefer them over the 300-line cell
        name = {"final_c": "c_topk"}.get(m.group(1), m.group(1))
        c = int(m.group(2))
        if c in cells[name] and cells[name][c]["n"] >= d["n"]:
            continue
        cells[name][c] = d

    levels = sorted({c for v in cells.values() for c in v})
    names = sorted(cells)

    print("## Profile x concurrency (TTFA p50 ms / audible-stall % / RTF; n in parentheses when 1088)\n")
    print("| profile | " + " | ".join(f"c={c}" for c in levels) + " |")
    print("|---|" + "---:|" * len(levels))
    for name in names:
        row = []
        for c in levels:
            d = cells[name].get(c)
            if d is None:
                row.append("–")
                continue
            tag = " (1088)" if d["n"] >= 1000 else ""
            row.append(f"{d['ttfa_p50_ms']:.0f} / {d['stall_100ms_pct']:.1f}% / {d['rtf_mean']:.2f}{tag}")
        print(f"| {name} | " + " | ".join(row) + " |")
    print("| H200 stock (#8933) | " + " | ".join(f"{H200.get(c, '–')} ms" if c in H200 else "–" for c in levels) + " |")

    print(f"\n## Best per concurrency (audible stalls <= {args.budget}%)\n")
    print("| c | best profile | TTFA p50 | TTFA p95 | RTF | strict / audible stalls | runner-up |")
    print("|---:|---|---:|---:|---:|---:|---|")
    for c in levels:
        ok = [(n, cells[n][c]) for n in names if c in cells[n] and cells[n][c]["stall_100ms_pct"] <= args.budget]
        if not ok:
            print(f"| {c} | (none within budget) | | | | | |")
            continue
        ok.sort(key=lambda t: (t[1]["ttfa_p50_ms"], t[1]["rtf_mean"]))
        n, d = ok[0]
        ru = f"{ok[1][0]} ({ok[1][1]['ttfa_p50_ms']:.0f} ms)" if len(ok) > 1 else "–"
        print(
            f"| {c} | **{n}** | {d['ttfa_p50_ms']:.0f} ms | {d['ttfa_p95_ms']:.0f} ms | {d['rtf_mean']:.3f}"
            f" | {d['stall_pct']:.1f} / {d['stall_100ms_pct']:.1f} % | {ru} |"
        )

    print("\n## One-deploy pick (admissible at every measured level)\n")
    cands = []
    for n in names:
        meas = [c for c in levels if c in cells[n]]
        if len(meas) < len(levels):
            continue
        if all(cells[n][c]["stall_100ms_pct"] <= args.budget for c in meas):
            cands.append((sum(cells[n][c]["ttfa_p50_ms"] for c in meas) / len(meas), n))
    if cands:
        cands.sort()
        for mean, n in cands:
            print(f"- {n}: mean TTFA p50 over levels {mean:.0f} ms")
    else:
        print("- none: every fully-measured profile breaks the stall budget at some level "
              "(this is the case for a load-adaptive ramp)")


if __name__ == "__main__":
    main()
