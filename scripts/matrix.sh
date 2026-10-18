#!/usr/bin/env bash
# Fill the profile x concurrency matrix (300-line subset per cell), skipping cells
# that already have a summary.json. One fresh server per profile.
# Usage: matrix.sh            (default plan below)   |   PLAN="stock:32 f_mid:1,4,16,32,64" matrix.sh
set -euo pipefail
cd ~/drama3
source .venv/bin/activate
DEFAULT_PLAN="stock:32 a_ramp:4,32,64 c_topk:4,64 e_ramp3:4,32,64 f_mid:1,4,16,32,64"
PLAN=${PLAN:-$DEFAULT_PLAN}
N=${N:-300}
declare -A YAML=( [stock]="" [a_ramp]=deploy/a_ramp.yaml [b_admission]=deploy/b_admission.yaml
                  [c_topk]=deploy/c_topk.yaml [e_ramp3]=deploy/e_ramp3.yaml [f_mid]=deploy/f_mid.yaml )
declare -A EXTRA=( [c_topk]="--top-k 100" [e_ramp3]="--top-k 100" [f_mid]="--top-k 100" )
for entry in $PLAN; do
  name=${entry%%:*}; levels=${entry#*:}
  todo=""
  for c in ${levels//,/ }; do [ -f "runs/${name}_c${c}/summary.json" ] || todo="$todo $c"; done
  [ -z "$todo" ] && { echo "== $name: all cells present"; continue; }
  echo "== $name: levels$todo"
  LEVELS="$todo" N=$N ./run_config.sh "$name" ${YAML[$name]:+$HOME/drama3/${YAML[$name]}} ${EXTRA[$name]:-}
done
tmux kill-session -t server 2>/dev/null || true
python best.py > runs/best.md
cat runs/best.md
echo MATRIX_DONE
