# 500k-model playing-strength regression — 2026-09-14

The two models have the **same 104,257 parameters** (781 → 128 → 32 → 1).
The larger experiment has more training positions, not a larger network or a fly
connectome. `fast-chess` uses an ordinary neural evaluator plus Python chess search.

## Findings

The 500k model improves held-out regression loss on the 500k dataset from 0.12569
(starter) to 0.08409. But its recorded 12-game match against the starter scored only
1 win, 1 draw and 10 losses. Regression loss did not predict playing strength.

The evidence points to an excessively influential learned correction to material:

* Across 200 reproducibly sampled corpus positions (NumPy seed 19), removing an
  enemy rook changes the starter's evaluation by a median 476 cp, but the 500k
  model's by only 369 cp. For a knight it is 282 versus 221 cp; a pawn, 96 versus
  68 cp. These artificial removal probes measure sensitivity, not ground-truth
  positional values.
* Training includes positions with checks, captures and recaptures still pending.
  The network is asked to predict Stockfish's searched evaluation directly from
  the current board. Search, however, uses the network after resolving captures
  and checks. A plausible mechanism is that the network learns to discount
  temporary material imbalances and carries that discount into search leaves.
* The conservative quiet-position filter retains 52,140 of the 500,000 positions.
  Retraining on those positions also improves the match result, supporting this
  explanation. It does not prove that every remaining position is tactically quiet.

Checks against alternative explanations:

* The quiet-data command verified feature and source-game alignment for **all
  500,000 rows** against `data/positions.tsv`.
* A fresh Stockfish check of 100 positions (seed 91, 10k nodes each) differed from
  the stored 50k-node labels by mean 23.19 cp, median 14 cp; none differed by more
  than 200 cp. This spot check supports label quality; it is not an exhaustive
  audit of every label or the remote shard history.
* Cached and from-scratch evaluation agreed within 0.00012 cp on the sampled
  positions and their first four legal children. This is float32 rounding noise.
* The original 31 tests passed before changes.

## Fix

`runs/teacher500k-calibrated/best.npz` uses:

```text
material + 0.5 × learned correction
```

The output-layer weights **and bias** are scaled, so the existing inference and
accumulator paths need no extra operations. Original weights and checkpoints remain
untouched. The exported artifact records the source model's SHA-256 and scale.
This scale was selected using games, and is specific to this model; it is not a
universal training default.

The calibrated artifact is now the shared default for `play`, `evaluate`, and
`benchmark`. Pass `--model runs/small/best.npz` to use the original starter.

Reproduce the export to a new path:

```bash
bash run.sh calibrate --model runs/teacher500k/best.npz \
  --scale .5 --out runs/reproduce-calibrated/best.npz
```

For future training, reuse the existing labels without paying for another
Stockfish run:

```bash
bash run.sh quiet --data data/teacher500k.npz --positions data/positions.tsv \
  --out data/reproduce-quiet.npz
bash run.sh train --data data/reproduce-quiet.npz \
  --out runs/reproduce-quiet --epochs 40 --lr .0003
```

The filter checks alignment before selecting rows, preserves source-game IDs for
validation splitting, and refuses to overwrite a destination. The verified output
matches the experimental quiet dataset exactly. Quiet-only training is an optional
alternative; the calibrated full-data model is the stronger development candidate.

## Development matches

Each experiment played 12 games against `runs/small/best.npz`, six paired openings,
0.3 seconds per move, depth cap 6, two worker processes, 300-ply cap. Experiments
overlapped on the same laptop, so these are candidate-selection results under CPU
contention, not a controlled Elo estimate.

| Model | Wins | Draws | Losses | Score |
|---|---:|---:|---:|---:|
| Full-data correction × 0.25 | 7 | 5 | 0 | 9.5/12 |
| Full-data correction × 0.50 | 8 | 3 | 1 | 9.5/12 |
| Quiet-data retraining | 6 | 4 | 2 | 8/12 |

All games finished. The 0.5 candidate was retained for confirmation; the small
sample does not distinguish the two scaling candidates reliably. Raw game records
are in `runs/scale025-vs-small.json`, `runs/scale050-vs-small.json`, and
`runs/quiet-vs-small.json`.

## Confirmation protocol

`tests/holdout-openings.json` contains six different openings, excluded from the
development matches. The scale is fixed before running these matches:

```bash
bash run.sh evaluate --model runs/teacher500k-calibrated/best.npz \
  --opponent model --rival-model runs/small/best.npz --games 12 \
  --seconds 1 --parallel 4 --max-plies 400 \
  --openings-file tests/holdout-openings.json \
  --out runs/calibrated-vs-small-holdout.json

bash run.sh evaluate --model runs/teacher500k-calibrated/best.npz \
  --opponent model --rival-model runs/teacher500k/best.npz --games 12 \
  --fixed-nodes --nodes 5000 --parallel 2 --max-plies 300 \
  --openings-file tests/holdout-openings.json \
  --out runs/calibrated-vs-original-fixed.json
```

The latter disables wall-clock limits, holding search work constant despite CPU
contention. Both retain the six-ply depth cap. New reports include both players'
mean depth and nodes, the openings, model hashes and source hashes. These tests
measure improvement against the two local models, not human or online Elo.

Equal-node confirmation against the original 500k model: **8 wins, 3 draws,
1 loss (9.5/12)**, all games finished. Mean completed depth per game averaged
3.43 for the calibrated model versus 3.97 for the original. The improvement thus
survived a fixed node cap and did not come from searching deeper. Runtime was
156 seconds with two workers, overlapping the clocked confirmation match.

Clocked confirmation against the starter: **8 wins, 3 draws, 1 loss (9.5/12)**,
all games finished, one second per move, four workers, 332 seconds elapsed. The
opening set was excluded from scale selection. This confirms the improvement at
a longer time budget as well as against the original full-data model. Both
confirmation matches ran on the same laptop and overlapped; the clocked result
reflects that contention. Those confirmation matches did not measure a Stockfish
or human Elo rating.

Validation: **35 tests passed**, including exported scaling, accumulator parity,
preservation of original weights, quiet-data alignment and special moves, and
fixed-node match reporting with paired custom openings. A starting-position
smoke test also exercised the playable export. The final default-path wiring
does not change search or evaluation arithmetic used in the confirmation games.

## Subsequent weakest-Stockfish benchmark

At the user's request, the calibrated model played 10 games against the installed
Stockfish 16 with `UCI_LimitStrength=true`, its minimum `UCI_Elo=1320`, and
`Skill Level=0`. Both sides received 60 seconds plus 0.6 seconds per move. Five
openings were paired with colors swapped, four games ran concurrently, and Fast
Chess retained its six-ply depth cap. No other benchmark experiments overlapped
this run.

**Result: 9 wins, 0 draws, 1 loss.** All ten games ended by checkmate; none were
unfinished or decided on time. Total runtime was 419 seconds. The nominal
Stockfish-relative performance calculation is approximately 1702 Elo, but ten
games against a stochastic, artificially weakened engine do not establish a
human or online rating.

```bash
bash run.sh benchmark --model runs/teacher500k-calibrated/best.npz \
  --games 10 --clock 60 --increment .6 --parallel 4 \
  --out runs/calibrated-stockfish-weakest-10
```

Use another output name when reproducing; the command preserves existing reports.
The full report is `runs/calibrated-stockfish-weakest-10.json`, and the ten verified
PGNs are in `runs/calibrated-stockfish-weakest-10.pgn`.
