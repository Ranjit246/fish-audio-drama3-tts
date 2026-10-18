#!/usr/bin/env bash
# Usage: run_config.sh <name> [deploy.yaml] [bench extra args...]
# One fresh server per config; each concurrency level gets its own 8 warmups.
# (Deviation from #8933, which restarted the server per level, to fit the time budget.)
set -euo pipefail
NAME=$1; shift
DEPLOY=${1:-}; [ $# -gt 0 ] && shift
LEVELS=${LEVELS:-"1 4 16 64"}
N=${N:-300}
cd ~/drama3
source .venv/bin/activate
~/drama3/serve.sh "$NAME" "$DEPLOY"
for c in $LEVELS; do
  OUT="runs/${NAME}_c${c}"; mkdir -p "$OUT"
  # 1 Hz GPU sampler for the run; mean/max utilization is folded into summary.json.
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$OUT/gpu.csv" &
  SMI=$!
  python bench.py --concurrency "$c" --n "$N" --out "$OUT" --save-wav "$@" 2>&1 | tail -1
  kill $SMI 2>/dev/null; wait $SMI 2>/dev/null || true
  python - "$OUT" <<'EOF'
import csv, json, sys
out = sys.argv[1]
rows = [r for r in csv.reader(open(f"{out}/gpu.csv")) if len(r) == 3]
util = [float(r[1]) for r in rows[2:]] or [0.0]   # skip the idle first samples
mem = [float(r[2]) for r in rows] or [0.0]
s = json.load(open(f"{out}/summary.json"))
s["gpu_util_mean_pct"] = sum(util) / len(util)
s["gpu_util_max_pct"] = max(util)
s["gpu_mem_max_mib"] = max(mem)
json.dump(s, open(f"{out}/summary.json", "w"), indent=2)
print(f"  gpu util mean {s['gpu_util_mean_pct']:.0f}% max {s['gpu_util_max_pct']:.0f}%  mem max {s['gpu_mem_max_mib']:.0f} MiB")
EOF
done
