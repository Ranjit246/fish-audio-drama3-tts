#!/usr/bin/env bash
# Final check: best config on ALL 1088 lines at c=1/16/32, then stop the server and
# score WER/SIM for stock vs best, then build the report.
# Usage: final.sh <name> <deploy.yaml> [bench extra args, e.g. --top-k 100]
set -euo pipefail
NAME=$1; DEPLOY=$2; shift 2
cd ~/drama3
source .venv/bin/activate
LEVELS="1 16 32" N=0 ./run_config.sh "$NAME" "$DEPLOY" "$@"
# free the GPU for scoring
tmux kill-session -t server 2>/dev/null || true
pkill -f "[v]llm serve" 2>/dev/null || true
for _ in $(seq 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 2; done
python score.py runs/stock_c1 runs/stock_c16 "runs/${NAME}_c1" "runs/${NAME}_c16" 2>&1 | grep -vE "Warning|warn" | tail -8
python report.py > runs/report.md
echo FINAL_DONE
