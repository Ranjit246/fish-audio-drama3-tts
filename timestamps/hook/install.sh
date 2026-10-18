#!/usr/bin/env bash
# Install (or remove) the Drama 3 alignment capture hook into a Python environment.
#
#   hook/install.sh [python]              # default: the `python` on PATH (activate the vllm venv first)
#   hook/install.sh --uninstall [python]
#
# Copies drama3_align_hook.py and drama3_align.pth into site-packages. The .pth
# makes every interpreter of that venv import the hook at startup; it stays
# inert unless DRAMA3_ALIGN_SWEEP=1 is in the environment (see README.md).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=install
if [ "${1:-}" = "--uninstall" ]; then MODE=uninstall; shift; fi
PY=${1:-python}
SP=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
if [ "$MODE" = uninstall ]; then
  rm -f "$SP/drama3_align_hook.py" "$SP/drama3_align.pth"
  echo "removed hook from $SP"
  exit 0
fi
cp "$HERE/drama3_align_hook.py" "$SP/drama3_align_hook.py"
cp "$HERE/drama3_align.pth" "$SP/drama3_align.pth"
echo "installed into $SP"
echo -n "inert check:  "; "$PY" -c 'import drama3_align_hook as h; print(h.STATUS)'
echo -n "armed check:  "; DRAMA3_ALIGN_SWEEP=1 DRAMA3_ALIGN_OUT=/tmp/drama3_align_check "$PY" -c 'import drama3_align_hook as h; print(h.STATUS)'
