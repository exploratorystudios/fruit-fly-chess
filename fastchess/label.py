"""Label sampled positions with a bounded Stockfish search, across many processes.

Labelling is the expensive stage and is pure CPU work: a GPU cannot help, because
the teacher is an alpha-beta engine. Throughput therefore scales with cores, so
this runs one single-threaded Stockfish per worker and writes per-worker shards
that let an interrupted run resume instead of starting over.
"""
import argparse
import json
import os
import shutil
import time
from multiprocessing import Process, Queue
from pathlib import Path
import chess
import chess.engine
import numpy as np
from .features import encode
from .positions import read


def label_shard(worker, rows, nodes, stockfish, shard_path, queue):
    """Label one worker's slice, appending to a resumable shard file."""
    done = {}
    if shard_path.exists():
        with open(shard_path) as handle:
            for line in handle:
                index, score = line.split()
                done[int(index)] = int(score)
    remaining = [(index, fen) for index, fen in rows if index not in done]
    queue.put(('resume', worker, len(done), len(remaining)))
    if remaining:
        with chess.engine.SimpleEngine.popen_uci(stockfish) as engine, \
                open(shard_path, 'a', buffering=1) as handle:
            engine.configure({'Threads': 1, 'Hash': 32})
            for count, (index, fen) in enumerate(remaining, 1):
                board = chess.Board(fen)
                info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
                score = info['score'].pov(board.turn).score(mate_score=10000)
                handle.write(f'{index} {score}\n')
                if count % 200 == 0:
                    queue.put(('progress', worker, count, len(remaining)))
    queue.put(('done', worker, 0, 0))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--positions', default='data/positions.tsv', help='File from `run.sh positions`')
    p.add_argument('--out', default='data/teacher.npz')
    p.add_argument('--nodes', type=int, default=10000)
    p.add_argument('--limit', type=int, default=0, help='Label only the first N positions; 0 means all')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)))
    p.add_argument('--shards', default='', help='Directory for resumable shards; defaults beside --out')
    p.add_argument('--stockfish', default=shutil.which('stockfish') or '/usr/games/stockfish')
    a = p.parse_args(argv)
    if a.nodes < 1 or a.workers < 1 or a.limit < 0:
        p.error('nodes and workers must be positive; limit must be nonnegative')
    if not shutil.which(a.stockfish) and not Path(a.stockfish).exists():
        p.error(f'Stockfish not found at {a.stockfish}; pass --stockfish')
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shards = Path(a.shards) if a.shards else out.with_suffix('.shards')
    shards.mkdir(parents=True, exist_ok=True)

    rows = read(a.positions)
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        p.error(f'No positions in {a.positions}')
    indexed = list(enumerate(rows))
    # Round-robin so every worker gets a similar mix of position types.
    slices = [[(index, fen) for index, (_, fen) in indexed[worker::a.workers]]
              for worker in range(a.workers)]
    print(f'{len(rows):,} positions, {a.nodes:,} nodes each, {a.workers} workers '
          f'({os.cpu_count()} cores visible)', flush=True)

    queue = Queue()
    workers = []
    start = time.monotonic()
    for worker, slice_rows in enumerate(slices):
        process = Process(target=label_shard, args=(worker, slice_rows, a.nodes, a.stockfish,
                                                    shards / f'shard-{worker:03d}.txt', queue))
        process.start()
        workers.append(process)
    progress = {worker: 0 for worker in range(a.workers)}
    totals = {worker: len(slices[worker]) for worker in range(a.workers)}
    resumed = finished = 0
    try:
        while finished < a.workers:
            kind, worker, value, total = queue.get()
            if kind == 'resume':
                resumed += value
                totals[worker] = total
            elif kind == 'progress':
                progress[worker] = value
                labelled = resumed + sum(progress.values())
                elapsed = time.monotonic() - start
                rate = sum(progress.values()) / max(elapsed, .001)
                left = (sum(totals.values()) - sum(progress.values())) / max(rate, .001)
                print(f'{labelled:,}/{len(rows):,} labelled, {rate:.0f}/s, '
                      f'{elapsed/60:.1f} min elapsed, ~{left/60:.0f} min left', flush=True)
            else:
                finished += 1
    except KeyboardInterrupt:
        print('Interrupted; shards are kept, re-run to resume.', flush=True)
        for process in workers:
            process.terminate()
        raise SystemExit(1)
    for process in workers:
        process.join()
        if process.exitcode:
            raise SystemExit(f'Worker exited with code {process.exitcode}; shards kept in {shards}')

    scores = {}
    for shard in sorted(shards.glob('shard-*.txt')):
        with open(shard) as handle:
            for line in handle:
                index, score = line.split()
                scores[int(index)] = int(score)
    order = sorted(scores)
    missing = len(rows) - len(order)
    features = np.stack([encode(chess.Board(rows[index][1])) for index in order])
    meta = dict(vars(a), positions_file=str(a.positions), labelled=len(order),
                missing=missing, seconds=time.monotonic() - start)
    tmp = out.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, X=features,
                        cp=np.array([scores[index] for index in order], dtype=np.int16),
                        game=np.array([rows[index][0] for index in order], dtype=np.int32),
                        metadata=json.dumps(meta))
    tmp.replace(out)
    print(f'Saved {len(order):,} labels to {out} in {meta["seconds"]/60:.1f} min'
          + (f' ({missing:,} missing)' if missing else ''))
    print(f'Shards kept in {shards}; delete them once you have the .npz.')


if __name__ == '__main__':
    main()
