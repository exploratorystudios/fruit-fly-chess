"""Sample human games and label positions with a bounded Stockfish search."""
import argparse
import json
import random
import shutil
import time
from pathlib import Path
import chess
import chess.engine
import chess.pgn
import numpy as np
from .features import encode


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pgn', default='../games/LumbrasGigaBase_OTB_2020-2024.pgn')
    p.add_argument('--out', default='data/teacher.npz')
    p.add_argument('--positions', type=int, default=50000)
    p.add_argument('--nodes', type=int, default=5000)
    p.add_argument('--per-game', type=int, default=16)
    p.add_argument('--skip-games', type=int, default=0)
    p.add_argument('--min-elo', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--stockfish', default=shutil.which('stockfish') or '/usr/games/stockfish')
    a = p.parse_args(argv)
    if min(a.positions, a.nodes, a.per_game) < 1 or a.skip_games < 0:
        p.error('positions, nodes and per-game must be positive; skip-games must be nonnegative')
    out = Path(a.out)
    if out.exists():
        p.error(f'{out} already exists; choose another --out')
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    xs, scores, groups, seen = [], [], [], set()
    start = time.monotonic()
    games = 0
    # Keeping duplicate positions in one partition avoids validation leakage.
    # Dedup key omits the move counters; stored features retain the halfmove clock.
    try:
        with chess.engine.SimpleEngine.popen_uci(a.stockfish) as engine, open(a.pgn, errors='replace') as fh:
            engine.configure({'Threads': 1, 'Hash': 32})
            while len(xs) < a.positions:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                games += 1
                if games <= a.skip_games or game.errors or game.headers.get('Variant', 'Standard') != 'Standard':
                    continue
                try:
                    if min(int(game.headers.get(k, 0)) for k in ('WhiteElo', 'BlackElo')) < a.min_elo:
                        continue
                except ValueError:
                    continue
                board = game.board()
                positions = []
                for ply, move in enumerate(game.mainline_moves()):
                    if not board.is_legal(move):
                        positions = []
                        break
                    if ply >= 8 and not board.is_game_over():
                        positions.append(board.copy(stack=False))
                    board.push(move)
                for board in rng.sample(positions, min(len(positions), a.per_game)):
                    key = ' '.join(board.fen().split()[:4])
                    if key in seen:
                        continue
                    info = engine.analyse(board, chess.engine.Limit(nodes=a.nodes), game=games)
                    cp = info['score'].pov(board.turn).score(mate_score=10000)
                    xs.append(encode(board))
                    scores.append(cp)
                    groups.append(games)
                    seen.add(key)
                    if len(xs) % 1000 == 0:
                        elapsed = time.monotonic() - start
                        print(f'{len(xs)}/{a.positions} positions, {len(xs)/elapsed:.0f}/s, {elapsed/60:.1f} min', flush=True)
                    if len(xs) >= a.positions:
                        break
    except KeyboardInterrupt:
        print('Interrupted; saving completed labels.', flush=True)
    if not xs:
        raise SystemExit('No usable positions found; check PGN and rating filter.')
    meta = dict(vars(a), games_read=games, actual_positions=len(xs), seconds=time.monotonic()-start)
    tmp = out.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, X=np.stack(xs), cp=np.array(scores, dtype=np.int16),
                        game=np.array(groups, dtype=np.int32), metadata=json.dumps(meta))
    tmp.replace(out)
    print(f'Saved {len(xs)} labels to {out} in {meta["seconds"]:.1f}s')


if __name__ == '__main__':
    main()
