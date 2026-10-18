#!/usr/bin/env bash
# Usage: serve.sh <name> [deploy.yaml]   (no yaml = stock fish_omni.yaml)
# Kills any running server, starts a fresh one in tmux "server", waits for /health.
set -euo pipefail
NAME=$1
DEPLOY=${2:-}
cd ~/drama3
source .venv/bin/activate
tmux kill-session -t server 2>/dev/null || true
pkill -f "[v]llm serve" 2>/dev/null || true
# vllm-omni starts a private MPS daemon per launch and leaves it behind; clear them
# or the GPU never reads as idle (and they hold 58 MiB each).
pkill -f "[n]vidia-cuda-mps-control" 2>/dev/null || true
pkill -f "[n]vidia-cuda-mps-server" 2>/dev/null || true
for _ in $(seq 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 2; done
EXTRA=""
[ -n "$DEPLOY" ] && EXTRA="--deploy-config $DEPLOY"
LOG=~/drama3/logs/server_${NAME}.log
# DeepGEMM JIT needs nvcc >= 13 for SM120 (system nvcc is 12.9); #8933 found Triton
# faster than DeepGEMM anyway. SERVE_ENV adds more "K=V" pairs per run.
ENV="VLLM_USE_DEEP_GEMM=0 ${SERVE_ENV:-}"
tmux new-session -d -s server "env $ENV vllm serve fishaudio/drama-3 --omni --port 8091 $EXTRA > $LOG 2>&1"
START=$(date +%s)
until curl -sf http://127.0.0.1:8091/health >/dev/null; do
  sleep 5
  if ! tmux has-session -t server 2>/dev/null; then echo "server died, see $LOG"; tail -30 $LOG; exit 1; fi
  if [ $(( $(date +%s) - START )) -gt 1500 ]; then echo "timeout"; tail -30 $LOG; exit 1; fi
done
echo "ready in $(( $(date +%s) - START ))s ($NAME)"
