# Local verification — September 14, 2026

Hardware: Intel Core i5-10300H (4 cores / 8 threads), GTX 1650 Ti (4 GB),
approximately 20 GB RAM. Other desktop applications remained running.

## Preparation and training

- Teacher: locally installed Stockfish 16, one thread, 32 MB hash, 5,000 nodes
  per position. PGN: the existing LumbrasGigaBase OTB 2020–2024 file.
- Dataset: 20,000 unique positions, maximum 16 samples per source game;
  prepared in **289.4 seconds**, approximately 69 positions/second.
- Split: 18,008 training positions and 1,992 validation positions, with source
  games kept in separate partitions. This validation set also selects the model;
  it is not an untouched final test set.
- Architecture: 781 → 128 → 32 → 1, **104,257 parameters**, learned correction
  to material. Exported playable weights occupy approximately **409 KiB**.
- Best CPU validation MSE: **0.09942**, versus **0.11599** for material alone
  (approximately **14.3% lower error** on the bounded evaluation target).
- Best validation MAE for teacher scores within ±1,000 cp: approximately **109 cp**.
- Best starter epoch: **7**. Later epochs overfit; the shipped `best.npz` remains
  epoch 7. The initial exploratory run recorded 40 epochs in `runs/small/metrics.json`.

After adding early stopping, separate fresh verification runs stopped at epoch 12:

| Device | Median training throughput | Total run time |
| --- | ---: | ---: |
| CPU, 2 threads | 117,261 positions/s | 2.90 s |
| GTX 1650 Ti, batch 512 | 183,166 positions/s | 1.79 s |

Throughput measures the batch training loop. Total time includes loading the tiny
dataset, model/optimizer setup, training, validation, and checkpoint writes, but
excludes initial Python/PyTorch imports. CPU was measured first in the same Python
process, so startup overhead differs; these are small-run feasibility measurements,
not a rigorous CPU/GPU comparison or a guarantee for larger runs. Raw results are
in `runs/training-benchmark.json`.

The original fly-chess README reports about 160 positions/s. Its architecture,
objective, and data pipeline differ, so a direct speedup ratio would not be a
controlled benchmark. The new setup clearly makes training practical; teacher
label preparation now accounts for most of the elapsed time.

## Chess checks

Nine automated checks passed: canonical color/state encoding, NumPy/PyTorch
prediction agreement, game-separated validation, mate finding, terminal positions,
timeout restoration, rook underpromotion to avoid stalemate, repetition, and
capturing a hanging queen. A separate smoke run verified partial training batches,
checkpoint export, and optimizer resume. Both actual CPU and GPU runs verified
early stopping and exported models.

The trained model selected mate in one in the supplied test position. From the
initial position it selected Nf3 and completed depth 4 at a one-second budget
(12,892 counted search/quiescence nodes in that run).

Four games against the same search using only material evaluation, at 0.1 seconds
per move and maximum depth 6, yielded **1 win / 1 draw / 2 losses**. Opening positions
were paired with colors swapped. No game reached the 240-ply cap. Full moves and
settings are in `runs/starter-vs-material.json`.

These results establish that the small network learns, runs, and plays legal games.
They do **not** establish that this starter is stronger than the material baseline,
nor do they establish Elo. Its initial playing strength is modest. More diverse
training positions, stronger teacher labels, and substantially longer matches are
needed before making a strength claim. The documented 50k-position experiment is
the next practical step; simply training the 20k slice longer made validation worse.


# Search rewrite — September 14, 2026

Same 104,257-parameter network and the same `runs/small/best.npz` weights throughout;
only `fastchess/search.py` changed. The previous search is frozen as
`fastchess/search_baseline.py` so the comparison can be re-run at any time.

Added: transposition table with depth/bound/move entries and ply-corrected mate
scores, null-move pruning (R=2), late move reductions, principal-variation search,
aspiration windows, check extensions, killer-move and history ordering, and delta
pruning in quiescence. Two plain speedups mattered as much as the pruning: keying the
table on `Board._transposition_key()` instead of `board_fen()` (36x cheaper), and
generating legal moves once per node instead of three times (the old `terminal()`
call generated them twice more before the move loop did).

## Search efficiency, depth 5, identical positions

| Position | Old nodes | New nodes | Old time | New time | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opening | 94,386 | 10,297 | 7.53 s | 0.70 s | 10.8x |
| Open middlegame | 286,899 | 59,069 | 21.71 s | 3.72 s | 5.8x |
| Tactical | 170,093 | 37,397 | 13.90 s | 2.33 s | 6.0x |
| Rook endgame | 29,854 | 5,716 | 2.01 s | 0.27 s | 7.4x |

Nodes per second also rose (about 12.5k to 15–21k) because of the two speedups above.
In the middlegame position the old search needed 21.7 s for depth 5; the new one
reaches **depth 7 in 20.8 s** — two extra plies at equal time.

## Head-to-head, new search vs frozen baseline

Twelve games, same network on both sides, equal 1 second per move, paired openings
with colours swapped, four games concurrently:

**7 wins / 4 draws / 1 loss = 9/12 (75.0%)** → **+191 Elo**, 95% interval [+47, +456].
Raw results in `runs/ab-search.json`.

The interval excludes zero, so the change is a real strength gain and not noise, but
twelve games pin it down only loosely. The point estimate should not be quoted alone.

## Ten games vs Stockfish at its weakest setting

Stockfish 16, `UCI_LimitStrength=true`, `UCI_Elo=1320` (the binary's minimum), Skill
Level 0, one thread, 32 MB hash. **Both engines shared the same 60+0.6 clock** — this
is an equal-time match, unlike the earlier untimed run. Four games ran concurrently.

**5 wins / 1 draw / 4 losses = 5.5/10 (55.0%)** → nominal performance **1355 Elo**,
95% interval [1136, 1611]. Mean search depth 4.16 plies. Every game ended in
checkmate or threefold repetition: no time forfeits and no games hit the 600-ply cap.
Raw results and PGN in `runs/v2-10.json` and `runs/v2-10.pgn`.

The whole match took **423 seconds**. The earlier untimed, unequal-time configuration
had completed 3 of 10 games in roughly 80 minutes before it was stopped.

Caveats that still apply: ten games is a small sample and the interval is wide; the
1320 anchor is Stockfish's own target for that setting, not a FIDE or online rating;
four concurrent games shared the laptop, so each engine had less CPU than in a solo
match; and `--parallel 1` would not reproduce this result exactly.

## Preserved results from the stopped untimed match

Three completed games before that run was stopped (Fast Chess untimed at depth 6,
Stockfish at 60+0.6 — an unequal-time comparison, and not combinable with the numbers
above): 2 wins and 1 loss, all by checkmate. Kept in
`runs/stockfish-weakest-10-untimed.json` and `runs/untimed-pairs/`.


# Incremental accumulator — September 14, 2026

Same network and weights again; only evaluation changed. Profiling the depth-6
middlegame search first showed where the time actually went:

| Part of one evaluation | Cost | Removed by the accumulator |
| --- | ---: | --- |
| `active_features(board)` | 18.6 us | yes |
| `w1[ids].sum(axis=0)` | 10.2 us | yes |
| `MATERIAL[ids].sum()` | 2.7 us | yes |
| small layers, clip, call overhead | ~32 us | no |
| **total** | **63.6 us** | |

Evaluation was 50% of search time (8.25 s of 16.4 s under the profiler), so it was
the right target. `np.clip` on a Python float also cost 3.9 us against 0.3 us for a
builtin clamp, which was simply a free fix.

The features are relative to the side to move, so a single running sum would be
invalidated every ply. `fastchess/accumulator.py` keeps **one accumulator per point
of view**, which makes each a pure function of the position and so incrementally
updatable; evaluation selects the view matching `board.turn`, and material is tracked
alongside because it is merely negated between views. Deltas come from XOR-ing the
twelve piece bitboards before and after the move, so castling, en passant and
promotion need no special cases. Both views share one `(2, width)` array with
pre-paired weight rows, halving the NumPy calls per changed feature — at this width
call overhead dominates the arithmetic.

## Effect on search speed

Measured back to back in one session, accumulator off versus on. Off is the same
`Search` handed a non-`Evaluator` callable, which disables the accumulator and
changes nothing else. **Node counts are identical in every position**, so this is
purely a speed change.

| Position (depth 6) | Off | On | Speedup | Nodes |
| --- | ---: | ---: | ---: | ---: |
| Opening | 1.64 s | 1.06 s | 1.54x | 24,683 |
| Middlegame | 9.90 s | 5.69 s | 1.74x | 121,959 |
| Tactical | 5.18 s | 3.45 s | 1.50x | 76,276 |
| Endgame | 0.89 s | 0.60 s | 1.48x | 15,953 |

Under the profiler, evaluation fell from 8.25 s to 1.21 s; the accumulator's own
bookkeeping costs 3.18 s of that back, for a clear net win. Across a six-position
suite at one second per move, nodes searched rose from 13,312 to 23,552 and mean
completed depth from 4.17 to 4.50.

Correctness rests on never drifting from a rebuild. `tests/test_accumulator.py` walks
random games with pushes and pops interleaved, comparing both view accumulators and
the final score against a from-scratch computation after every move and every unwind,
with castling, en passant and promotion oversampled and also covered by named cases.
Worst observed difference across ~22,000 positions was 9.5e-05 centipawns, which is
float32 accumulation noise; the material-only path is exact.

## Head-to-head, accumulator on vs off

Twelve games, identical search and network on both sides, equal 1 second per move,
paired openings with colours swapped:

**8 wins / 3 draws / 1 loss = 9.5/12 (79.2%)** -> **+232 Elo**, 95% interval
[+78, +621]. Raw results in `runs/ab-accumulator.json`.

That is a large return on a ~1.5x speedup, but it is consistent: at four to five
plies each additional ply is worth far more than in a deep engine. The interval is
wide and excludes zero.

## Ten games vs Stockfish 1320, repeated

Same configuration as before — equal 60+0.6 clock, four games concurrently:

| Build | Score | Performance Elo | 95% interval | Mean depth | Wall clock |
| --- | ---: | ---: | --- | ---: | ---: |
| Step 1 (search rewrite) | 5.5/10 | 1355 | [1136, 1611] | 4.16 | 423 s |
| Step 2 (+ accumulator) | **8.5/10** | **1621** | [1429, unbounded] | 4.52 | 416 s |

Results in `runs/v3-10.json` and `runs/v3-10.pgn`.

These are two independent ten-game samples, so the three-point jump is **not** a clean
measurement of the accumulator: a swing that large from 0.36 extra plies is partly
noise. The twelve-game head-to-head above is the number to trust for attribution;
the Stockfish matches show where the engine now stands overall.

The 95% upper bound at 8.5/10 reaches a 100% score, where no finite Elo exists. The
report now says so explicitly instead of emitting a bare null, which previously read
like a failed computation.

## Where this leaves the engine

Measured in games rather than nodes, the two changes are worth +191 and +232 Elo in
their own head-to-head matches. Playing strength is still bounded by search depth
rather than evaluation quality: the engine averages 4.5 plies at 60+0.6, and the
remaining losses are tactical. More training data (step 3) is the next lever, and the
per-view accumulator is also the structure that king-bucketed features (step 4) need.
