"""Collect runs/*/summary.json (+ score.json when present) into one markdown table,
next to the H200 reference numbers from vllm-omni PR #8933 (Seed-TTS en, ref_audio_codes).

Usage: report.py [runs_dir] > report.md
"""

import json
import sys
from pathlib import Path

H200 = {  # concurrency: (ttfa_p50_ms, ttfa_p95_ms, rtf, frames_per_s)
    1: (66, 70, 0.135, 161),
    2: (70, 84, 0.141, 308),
    4: (75, 93, 0.157, 552),
    8: (82, 108, 0.183, 949),
    16: (104, 139, 0.239, 1428),
    32: (135, 176, 0.318, 2134),
    64: (198, 396, 0.375, 3495),
    128: (255, 657, 0.515, 4835),
}


def f(x, nd=0):
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def main() -> None:
    runs = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "drama3/runs")
    rows = []
    for d in sorted(runs.iterdir()):
        s = d / "summary.json"
        if not s.exists() or d.name == "smoke":
            continue
        summ = json.loads(s.read_text())
        score = json.loads((d / "score.json").read_text()) if (d / "score.json").exists() else {}
        rows.append((d.name, summ, score))

    print("| run | mode | c | n | TTFA p50 | TTFA p95 | RTF | audio s/s | frames/s | stall % | runaway | WER % | WER % (no runaway) | SIM |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, s, sc in rows:
        print(
            f"| {name} | {s['mode']}{' top_k=' + str(s['top_k']) if s.get('top_k') else ''} | {s['concurrency']} | {s['n']}"
            f" | {f(s['ttfa_p50_ms'])} ms | {f(s['ttfa_p95_ms'])} ms | {f(s['rtf_mean'], 3)} | {f(s['audio_s_per_s'], 1)}"
            f" | {f(s['frames_per_s'])} | {f(s['stall_pct'], 1)} | {s['runaway']}"
            f" | {f(sc.get('wer_mean_pct'), 2)} | {f(sc.get('wer_mean_no_runaway_pct'), 2)} | {f(sc.get('sim_mean'), 3)} |"
        )
    print()
    print("H200 reference (PR #8933, 1088 lines, fresh server per level):")
    print()
    print("| c | TTFA p50 | TTFA p95 | RTF | frames/s |")
    print("|---:|---:|---:|---:|---:|")
    for c, (p50, p95, rtf, fps) in H200.items():
        print(f"| {c} | {p50} ms | {p95} ms | {rtf:.3f} | {fps} |")


if __name__ == "__main__":
    main()
