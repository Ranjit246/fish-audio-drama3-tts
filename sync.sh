#!/usr/bin/env bash
# Pull benchmark results + logs from the VM (no WAVs), push local script/deploy edits up.
set -euo pipefail
# Set DRAMA3_VM=user@host (the GPU box); the remote layout is ~/drama3/{runs,logs,...}.
VM=${DRAMA3_VM:?set DRAMA3_VM=user@host}
HERE=$(cd "$(dirname "$0")" && pwd)
rsync -az --include='*/' --include='summary.json' --include='score.json' --include='results.jsonl' --include='gpu.csv' --include='score_items.jsonl' --exclude='*' $VM:drama3/runs/ "$HERE/results/"
rsync -az --exclude='*.csv' $VM:drama3/logs/ "$HERE/logs/"
rsync -az $VM:drama3/manifest.jsonl "$HERE/results/manifest.jsonl"
if [ "${1:-}" = "--push" ]; then
  rsync -az "$HERE/scripts/" $VM:drama3/
  rsync -az "$HERE/deploy/" $VM:drama3/deploy/
fi
echo "synced $(date)"
