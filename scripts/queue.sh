#!/usr/bin/env bash
# GPU work queue: wait for matrix.sh (runs/best.md), validate the load-adaptive ramp
# (patched vllm_omni via PYTHONPATH shadowing), then run the attention head sweep.
set -uo pipefail
cd ~/drama3
source .venv/bin/activate
export PATH=$HOME/.local/bin:$PATH
until [ -f runs/best.md ]; do sleep 30; done
echo "== matrix finished; adaptive-ramp validation"
DEV=$HOME/drama3/vllm-omni-dev
SERVE_ENV="PYTHONPATH=$DEV" LEVELS="1 16 32 64" ./run_config.sh adaptive "$DEV/vllm_omni/deploy/fish_omni_low_latency.yaml" --top-k 100 2>&1 | tail -12
grep -E "ramp|Ramp|window" logs/server_adaptive.log | sed -e "s/\x1b\[[0-9;]*m//g" | head -5
python best.py > runs/best.md
echo ADAPTIVE_DONE
echo "== head sweep"
SERVE_ENV="DRAMA3_ALIGN_SWEEP=1 DRAMA3_ALIGN_OUT=$HOME/drama3/align_capture DRAMA3_ALIGN_IDLE_S=5" ./serve.sh sweep timestamps/sweep_deploy.yaml 2>&1 | tail -2
python timestamps/sweep/run_sweep.py --manifest manifest.jsonl --out timestamps/sweep/out 2>&1 | tail -5
tmux kill-session -t server 2>/dev/null; pkill -f "[v]llm serve" 2>/dev/null; sleep 15
ls align_capture | head; ls align_capture | wc -l
python timestamps/sweep/match.py --capture-dir ~/drama3/align_capture --requests timestamps/sweep/out/requests.jsonl 2>&1 | tail -8
python timestamps/sweep/score_heads.py --capture-dir ~/drama3/align_capture --requests timestamps/sweep/out/requests.jsonl --topk 6 2>&1 | tail -25
REQ=$(ls align_capture/req_*.npz | sed -n 3p)
python timestamps/decode/align_decode.py --npz "$REQ" --requests timestamps/sweep/out/requests.jsonl --heads timestamps/sweep/out/alignment_heads.json 2>&1 | tail -30
echo SWEEP_DONE
