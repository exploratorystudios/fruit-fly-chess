# Fast Chess

A small chess evaluation network you can train on this laptop, with its own
CPU chess search. The original `fly-chess` is untouched. This version replaces
the fly connectome with an ordinary feed-forward network.

The network has **104,257 trainable parameters**: 781 inputs → 128 → 32 → 1.
It learns a correction to material evaluation from Stockfish-labeled positions.
Training defaults to two CPU threads; the GTX 1650 Ti is optional. Playing uses
NumPy and python-chess, without running Stockfish or importing PyTorch.

This is a practical small-engine starting point, **not a claim of master strength**.
A quick starter run cannot establish Elo. More varied teacher data and measured
matches are the next steps; validation loss alone is not playing strength.

## 500k model correction

The 500k-data model has the same network size as the starter. Its lower validation
loss hid a playing-strength regression: its learned correction discounted material
too much. The corrected export keeps the full-data network and halves its correction
to material, with no extra inference cost. It is now the default for play, evaluate
and benchmark. On six new paired openings it beat the starter 8 wins, 3 draws,
1 loss at one second per move:

```bash
bash run.sh play --model runs/teacher500k-calibrated/best.npz --seconds 1
```

Original weights remain in `runs/teacher500k/best.npz`. The new `calibrate` command
exports a scaled copy with provenance; the `quiet` command prepares a training
subset from existing labels after verifying every row against the original FEN
file. See [the diagnosis, experiments, and reproduction commands](DIAGNOSIS.md).
The chosen scale is specific to this checkpoint; measure new models in games
instead of assuming the same scale or the lowest validation loss will win.

## Start here

Run these commands from this folder. `run.sh` reuses the working Python environment
in `../fly-chess/.venv` unless this folder has its own `.venv`.

```bash
cd /home/thewindmage/Documents/chess-ccg/fast-chess

# Play the corrected model. Enter e4 or e2e4; quit and undo also work.
bash run.sh play --seconds 1

# Next useful experiment: more examples, followed by a fresh training run.
bash run.sh data --positions 50000 --nodes 5000 --out data/teacher.npz
bash run.sh train --data data/teacher.npz --out runs/teacher50k --epochs 30

# Resume an interrupted experiment; optionally switch to the GPU.
bash run.sh train --data data/teacher.npz --out runs/teacher50k --resume --device cuda
```

`runs/teacher500k-calibrated/best.npz` is the default for playing, selected in games.
`runs/small/best.npz` retains the original starter for comparisons. Training's
`best.npz` is selected by validation loss and should be tested in games separately.
`last.pt` contains model and optimizer state for resume; `last.npz` is the latest
playable model. `metrics.json` records each epoch. Ctrl+C during training validates
and saves progress; the time budget is checked each batch, followed by validation
and checkpoint writing. An interrupted partial epoch resumes with a fresh shuffle.
Training stops after five epochs without validation improvement (`--patience 0`
disables this). The included starter already overfits with additional epochs;
adding examples is more useful than repeatedly resuming it.

On this computer, the verified early-stopped 20k-position run took **2.90 seconds
on CPU / 1.79 seconds on GPU**, after **289 seconds of label preparation**.
With the rewritten search, the historical starter model scores **5.5/10 against Stockfish 16 at
its weakest setting** (UCI_Elo 1320) on an equal 60+0.6 clock — a nominal 1355
performance rating, 95% interval [1136, 1611]. The search rewrite itself is worth
**+191 Elo** head-to-head against the previous search with the same network
(9/12, 95% interval [+47, +456]). See [measured results and limitations](RESULTS.md).

Both evaluator quality and search depth limit playing strength. The starter
averages about 4 plies at that time control; the later 500k experiment also showed
that a better validation loss can accompany much worse tactical play.

## Train a larger model dataset

Stockfish is the teacher during **data preparation only**. The existing PGN is reused;
no game database download is needed. A node budget limits work for each label.

```bash
# About 50k labels; measure preparation time separately from training time.
bash run.sh data --positions 50000 --nodes 5000 --out data/teacher.npz
bash run.sh train --data data/teacher.npz --out runs/teacher50k --epochs 30 --minutes 10
bash run.sh play --model runs/teacher50k/best.npz --seconds 2

# A longer experiment, with stronger teacher labels and more varied positions.
bash run.sh data --positions 200000 --nodes 20000 --out data/teacher200k.npz
bash run.sh train --data data/teacher200k.npz --out runs/teacher200k --epochs 30 --minutes 30
```

Start with 50k and compare results before paying for the larger run. Teacher
generation is the expensive step, and it happens only once per dataset. Increasing
the teacher node budget increases preparation time. Use `--skip-games` to sample
later games in the PGN. For a different PGN, pass `--pgn path/to/games.pgn`;
`--min-elo 0` disables the default 2000+ rating filter.

The builder samples up to 16 positions per game, skips the first eight plies and
malformed games, and removes duplicate FEN positions (ignoring move counters).
Validation reserves approximately 10% of source games, rather than randomly
mixing nearby positions from the same game between training and validation.
Do not compare validation numbers across different datasets as if they were the
same test. Resume requires the same dataset path and seed. Use a new `--out`
when changing datasets; the current trainer starts that experiment from scratch.

## Label a large dataset on Kaggle

Labelling is the only expensive stage, and it is **pure CPU work** — the teacher is
Stockfish, an alpha-beta search. GPUs cannot accelerate it, so throughput scales with
cores and nothing else. Sampling positions from the PGN is cheap and stays local;
only labelling is worth sending to a bigger machine.

```bash
# 1. Sample locally (seconds, no engine involved).
bash run.sh positions --positions 500000 --out data/positions.tsv

# 2. Bundle the positions and the fastchess package for upload.
bash kaggle/package.sh data/positions.tsv

# 3. Upload and run (prints the exact commands, including the notebook push).
pip install kaggle
kaggle datasets create -p kaggle/upload --dir-mode zip
cp kaggle/kernel-metadata.local.json kaggle/kernel-metadata.json
kaggle kernels push -p kaggle

# 4. Download the labels and retrain locally.
kaggle kernels output <username>/fast-chess-label-and-train -p data/
bash run.sh train --data data/teacher.npz --out runs/teacher500k --epochs 60
```

`kernel-metadata.json` ships with `enable_gpu: false` on purpose: a GPU session
labels no faster and gets a shorter time limit. The notebook sizes its worker pool to
whatever cores the session reports, and writes resumable shards, so a session that
times out can be re-run and picks up where it stopped.

The same two stages work locally, which is also how the Kaggle path is tested:

```bash
bash run.sh positions --positions 50000 --out data/positions.tsv
bash run.sh label --positions data/positions.tsv --out data/teacher.npz --nodes 10000
```

`label` defaults to one worker per core, one single-threaded Stockfish each. On this
laptop that is roughly 37 positions/s per worker at 10,000 nodes. The older one-shot
`run.sh data` command still works and remains the simplest local path.

## Play it in a browser

```bash
bash run.sh site                       # then open http://127.0.0.1:8000/
bash run.sh site --port 9000 --seconds 4 --model runs/small/best.npz
```

A local page with a large board (720px on a wide screen, pieces scaled to the
squares via container queries) that plays the engine and shows what it is
computing while it plays:
the live evaluation split into material and network correction, both hidden layers
firing per position, the twelve piece-input planes as the network sees them (from
the mover's point of view, "us" and "them"), the search's depth, node count and
expected line, and every legal move scored by the network alone with no look-ahead.

That last panel is the interesting one: the top of the static list is often *not*
the move the search plays, which is the clearest demonstration that the strength
lives in the search rather than the network.

**3D mind** opens a collapsible side panel showing the whole network as four layers
in space: the input features switched on in this position, the 128 and 32 hidden
neurons (height and brightness are activation strength), and the single output.
The lines between layers are the real `w2` and `w3` weights — green for positive,
red for negative — drawn only where both ends are active, strongest 900 of 4,096.
Drag to rotate, scroll to zoom, and toggle auto-rotate, connections or inputs. The
panel remembers whether it was open.

It is drawn with a hand-rolled projection on a 2D canvas rather than a 3D library,
so the page still needs no build step, no CDN and no network at all.

The page talks to the real `fastchess` engine over a small JSON API, so the picture
can never drift from the network that is actually choosing moves — `Evaluator.inspect`
returns the intermediate values and a test asserts it agrees with `Evaluator.__call__`.
Stdlib only, no build step and no internet; it binds to localhost.


## The actual fly brain

There are two networks in this repo and they are not the same thing. The one that
plays well is the 104,257-parameter evaluator above. The other is the reason the
project is called what it is: a spiking model built on a **6,000-neuron subgraph of
the FlyWire connectome**, the complete adult *Drosophila* brain. It has no search at
all — 773 board features drive 1,024 sensory neurons, the connectome runs for 32
timesteps of leaky integrate-and-fire dynamics, and the 1,024 highest in-degree
neurons are read out as a score for every from/to square pair.

```bash
bash run.sh site --fly            # then open http://127.0.0.1:8137/fly.html
```

`/fly.html` plays that model and draws it **where it actually is**. Every neuron sits
at its measured FlyWire position, joined by root id from the public v783 archive, so
the shape on screen is the fly's brain rather than an invented layout — the central
mass, the optic lobe and the tracts between them are all real. Synapse endpoints,
Dale's-law signs and the learned per-synapse gains come from the trained checkpoint,
and the neurons that light up are the ones that genuinely fired: the full
6,000 x 32 spike raster is bit-packed and sent with each move, so the animation is
the network's own activity, not a re-simulation or a summary.

Colour by role (sensory, motor, interneuron) or by neuropil, scrub the 32 timesteps
by hand, and read the move scores straight off the readout.

Be clear-eyed about its chess: it predicts the played move in 30.3% of held-out
positions, 57.9% in its top five. It is far weaker than the small evaluator, and it
is meant to be interesting rather than strong.

The weights ship in `model/fly.pt` and the anatomy asset in `public/fly/` (188 KiB
for 6,000 neurons and the 12,000 strongest of 697,316 synapses). Regenerate the
asset with:

```bash
python -m fastchess.fly_export --coordinates path/to/coordinates.csv.gz
```

This mode needs PyTorch, so it runs locally only and is excluded from the Vercel
deployment.

## Deploying to Vercel

`app.py` is a plain WSGI application — Vercel's Python runtime loads one top-level
`app` and routes every request to it, so this serves both the static page and the
engine API. The API calls the same `dispatch()` the local server uses, so the
deployment cannot drift from `bash run.sh site`.

```bash
npm i -g vercel
vercel            # preview
vercel --prod
```

No web framework is used on purpose: the function bundle is just numpy and
python-chess, which keeps cold starts short. Everything the deployment needs is in
the repo — the engine (`fastchess/`), the weights (`model/best.npz`, 410 KB) and the
page (`public/index.html`). Training data, PGNs and experiment runs are excluded via
`.gitignore` and `.vercelignore`; PyTorch lives in `requirements-dev.txt` and is
needed only for training, never for playing.

`vercel.json` gives the function 1 GB and a 15-second ceiling (inside the Hobby
limit; raise it on a paid plan). Thinking time is
clamped server-side so a slow search returns a move rather than a timeout:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLY_SECONDS` | `1.5` | Default thinking time per move |
| `FLY_DEPTH` | `8` | Maximum search depth |
| `FLY_MAX_SECONDS` | `4` | Hard cap on a single request's search |

A cold start pays for importing numpy and loading the weights; the engine is then
cached per warm container, so the first move after an idle period is slower.

## Measure playing strength

```bash
# Ten games, paired openings with colors swapped; may take several minutes.
bash run.sh evaluate --games 10 --seconds .1 --opponent material

# A much harder opponent: Stockfish at a fixed node budget per move.
bash run.sh evaluate --games 10 --seconds .2 --opponent stockfish --stockfish-nodes 1000 --out runs/vs-stockfish.json

# Analyze a single position, with score from the side-to-move perspective.
bash run.sh play --fen '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'
bash run.sh test
```

The material opponent uses exactly the same search without the learned correction,
so it is useful for checking whether a training change helps. Ten games are a smoke
test, not a statistically reliable rating. Games hitting the move cap are explicitly
reported as unfinished, not silently awarded draws. Results include moves and final
positions. A fixed-node Stockfish opponent does not correspond to a known Elo.

For a nominal rating comparison against the installed Stockfish's weakest setting:

```bash
# Ten games, four at a time. Both engines share the 60+0.6 clock.
bash run.sh benchmark --games 10 --clock 60 --increment .6 --parallel 4 --out runs/match

# Quick signal while iterating (weaker play at a short clock, but minutes not hours).
bash run.sh benchmark --games 10 --clock 10 --increment .1 --parallel 4 --out runs/quick

# Harder opponent, to find the ceiling once the weakest setting is beaten.
bash run.sh benchmark --games 10 --opponent-elo 1800 --parallel 4 --out runs/elo1800

# Give Fast Chess unlimited time to complete its six-ply search on every move.
bash run.sh benchmark --games 10 --untimed-fast --parallel 4 --out runs/untimed-match
```

Games are independent, so `--parallel` runs several at once; that is what turns a
multi-hour match into a few minutes. Use `--parallel 1` for a strictly solo match.
A parallel run shares the CPU between games, which the report records alongside the
score, because a clocked result is not identical to a one-game-at-a-time run.

This uses `UCI_LimitStrength=true` and, by default, the binary's minimum `UCI_Elo`
(1320 for the installed Stockfish 16) with Skill Level 0; `--opponent-elo` raises it.
Clocked mode gives both engines the same clock; untimed mode gives Fast Chess no time
or node limit while preserving its configured search depth (including quiescence), and
those games take far longer. Outputs include per-move timings and depths, model and
source hashes, PGN games, and the score's nominal Elo performance relative to
Stockfish's target. Zero points gives no finite Elo point estimate, so the report
supplies a one-sided upper bound. Ten games cannot establish an official human or
online rating, and an untimed match is not an equal-time comparison at all.

## Design and limitations

- Features: 12 piece planes from the mover's perspective, four castling rights,
  eight en-passant file indicators, and the halfmove clock. All promotion choices
  remain distinct legal moves in search.
- Objective: mean squared error between `tanh(predicted_cp / 400)` and the teacher
  target. The material baseline is added outside the learned layers. Game results
  are not used as substitutes for position evaluations.
- Inference keeps the first layer in an incrementally updated accumulator
  (`accumulator.py`) instead of rebuilding it at every leaf. Because the features
  are relative to the side to move, one running sum per point of view is kept; each
  is then a pure function of the position, so a move only adds and removes the few
  features it changed, and evaluation selects the view matching `board.turn`.
  Material is tracked the same way. Both views share one array with pre-paired
  weight rows, since at this width NumPy call overhead dominates the arithmetic.
  This is the accumulator idea used by NNUE evaluators; it is still **not** a full
  quantized Stockfish NNUE, and there are no king-bucketed features yet.
- The accumulator only stays correct while every push and pop is mirrored, so all
  search pushes go through `Search.push`/`Search.pop`. Passing any callable that is
  not an `Evaluator` disables it and falls back to from-scratch evaluation, which is
  also how the tests A/B the two paths.
- Search: iterative deepening alpha-beta negamax with a transposition table,
  null-move pruning, late move reductions, principal-variation search, aspiration
  windows, check extensions, killer-move and history ordering on top of MVV-LVA,
  and capture/promotion quiescence with delta pruning. All evasions are searched
  when checked. The search restores the board even on timeout, and a hard
  quiescence limit prevents pathological checking sequences. Very tiny time/node
  budgets can return the first legal move if no iteration completes.
- Transposition entries are cleared at the start of every search and are not
  trusted for score cutoffs at the root or within ten halfmoves of the fifty-move
  limit, so a cached score cannot outlive the draw history that justified it.
  `search_baseline.py` freezes the pre-optimisation search purely for A/B tests.
- The table is keyed on python-chess's `Board._transposition_key()`, a private API
  chosen because it is ~36x cheaper than `board_fen()`. It is pinned by the
  `python-chess>=1.999,<2` bound in `requirements.txt`; if a future release drops it,
  substitute `chess.polyglot.zobrist_hash` and expect the search to slow down.
- This Python search will be much slower and shallower than a native engine.
  The small training slice also underrepresents unusual endgames. There are no
  tablebases, opening book, UCI integration, or browser UI in this version.
- Draws already available by repetition/50 moves are treated as drawn in search.
  The terminal player and match runner claim available draws, including claims
  available by announcing the next move.

The broad neural-evaluation-plus-search approach is described in the
[Stockfish NNUE introduction](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/).
This implementation and its weights are independent of Stockfish's network.

## Separate environment (optional)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Data preparation also needs a Stockfish binary (already present at
`/usr/games/stockfish` on this computer); pass `--stockfish /path/to/stockfish`
if necessary. The tested local environment is Python with PyTorch 2.5.1+cu121,
NumPy 2.5.3 and chess 1.11.2. The broad dependency bounds are not a reproducibility
lockfile; the existing environment avoids extra downloads.
