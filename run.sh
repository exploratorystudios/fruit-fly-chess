#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Prefer an isolated environment; reuse the existing working installation otherwise.
if [[ -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
elif [[ -x ../fly-chess/.venv/bin/python ]]; then
  python_bin=../fly-chess/.venv/bin/python
else
  python_bin=python3
fi
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
  data|positions|label|quiet|calibrate|train|play|evaluate|benchmark) exec "$python_bin" -m "fastchess.$command" "$@" ;;
  site) exec "$python_bin" -m fastchess.site_server "$@" ;;
  test) exec "$python_bin" -m unittest discover -s tests -v "$@" ;;
  *) echo 'Usage: bash run.sh {data|positions|label|quiet|calibrate|train|play|evaluate|benchmark|site|test} [options]'; exit 0 ;;
esac
