#!/usr/bin/env bash
# Build the Kaggle upload bundle: sampled positions + the fastchess package, so
# the notebook needs no repository checkout. Sampling runs here because it is
# cheap; only the Stockfish labelling is worth sending to a bigger machine.
set -euo pipefail
cd "$(dirname "$0")/.."

USER="${KAGGLE_USERNAME:-USERNAME}"
OUT=kaggle/upload
POSITIONS="${1:-data/positions.tsv}"

if [ ! -f "$POSITIONS" ]; then
  cat >&2 <<EOF
No position file at $POSITIONS. Create one first, for example:
  bash run.sh positions --positions 500000 --out data/positions.tsv
Then re-run: bash kaggle/package.sh [path/to/positions.tsv]
EOF
  exit 1
fi

rm -rf "$OUT" && mkdir -p "$OUT/fastchess"
cp fastchess/*.py "$OUT/fastchess/"
gzip -9 -c "$POSITIONS" > "$OUT/positions.tsv.gz"
sed "s/USERNAME/$USER/g" kaggle/dataset-metadata.json > "$OUT/dataset-metadata.json"
sed "s/USERNAME/$USER/g" kaggle/kernel-metadata.json > kaggle/kernel-metadata.local.json

lines=$(( $(wc -l < "$POSITIONS") - 1 ))
echo "positions: $POSITIONS ($lines rows)"
echo "bundle ready: $OUT ($(du -sh "$OUT" | cut -f1))"
cat <<EOF

Next:
  pip install kaggle                       # token in ~/.kaggle/kaggle.json
  kaggle datasets create -p $OUT --dir-mode zip
  # later updates:  kaggle datasets version -p $OUT -m "more positions" --dir-mode zip

Then push the notebook:
  cp kaggle/kernel-metadata.local.json kaggle/kernel-metadata.json
  kaggle kernels push -p kaggle

When the run finishes, download its output and retrain locally:
  kaggle kernels output $USER/fast-chess-label-and-train -p data/
  bash run.sh train --data data/teacher.npz --out runs/teacher500k --epochs 60
EOF
