"""Sample positions from a PGN without running any engine.

Splitting sampling from labelling is what makes remote labelling practical: the
position list is small and cheap to produce, while labelling is the expensive part
and is embarrassingly parallel once the positions are fixed. Sampling here rather
than in each worker also keeps the set deterministic and already deduplicated, so
workers never need to coordinate.
"""
import argparse
import random
import sys
from pathlib import Path
import chess
import chess.pgn

OPENING_PLIES = 8


def dedup_key(board):
    """FEN without the move counters: the same position reached by a different
    move order is one position, but the halfmove clock is still kept in features."""
    return ' '.join(board.fen().split()[:4])


def sample(pgn_path, count, per_game=16, min_elo=2000, skip_games=0, seed=42, progress=None):
    """Yield (game_index, fen) pairs, deduplicated, until `count` are produced."""
    rng = random.Random(seed)
    seen = set()
    produced = games = 0
    with open(pgn_path, errors='replace') as handle:
        while produced < count:
            offset = handle.tell()
            headers = chess.pgn.read_headers(handle)
            if headers is None:
                break
            games += 1
            if games <= skip_games or headers.get('Variant', 'Standard') != 'Standard':
                continue
            try:
                if min(int(headers.get(key, 0)) for key in ('WhiteElo', 'BlackElo')) < min_elo:
                    continue
            except ValueError:
                continue
            # Only games that pass the cheap header filters are parsed in full.
            handle.seek(offset)
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            if game.errors:
                continue
            board = game.board()
            positions = []
            for ply, move in enumerate(game.mainline_moves()):
                if not board.is_legal(move):
                    positions = []
                    break
                if ply >= OPENING_PLIES and not board.is_game_over():
                    positions.append(board.copy(stack=False))
                board.push(move)
            for chosen in rng.sample(positions, min(len(positions), per_game)):
                key = dedup_key(chosen)
                if key in seen:
                    continue
                seen.add(key)
                yield games, chosen.fen()
                produced += 1
                if progress and produced % progress == 0:
                    print(f'{produced}/{count} positions from {games} games', flush=True)
                if produced >= count:
                    break


def read(path):
    """Read back a sampled file as a list of (game_index, fen)."""
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            game, fen = line.split('\t', 1)
            rows.append((int(game), fen))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pgn', default='../games/LumbrasGigaBase_OTB_2020-2024.pgn')
    p.add_argument('--out', default='data/positions.tsv')
    p.add_argument('--positions', type=int, default=500000)
    p.add_argument('--per-game', type=int, default=16)
    p.add_argument('--skip-games', type=int, default=0)
    p.add_argument('--min-elo', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args(argv)
    if min(a.positions, a.per_game) < 1 or a.skip_games < 0:
        p.error('positions and per-game must be positive; skip-games must be nonnegative')
    out = Path(a.out)
    if out.exists():
        p.error(f'{out} already exists; choose another --out')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    written = 0
    with open(tmp, 'w') as handle:
        handle.write(f'# fastchess positions\tseed={a.seed}\tper_game={a.per_game}\t'
                     f'min_elo={a.min_elo}\tskip_games={a.skip_games}\tpgn={Path(a.pgn).name}\n')
        for game, fen in sample(a.pgn, a.positions, a.per_game, a.min_elo,
                                a.skip_games, a.seed, progress=25000):
            handle.write(f'{game}\t{fen}\n')
            written += 1
    if not written:
        tmp.unlink(missing_ok=True)
        sys.exit('No usable positions found; check the PGN path and the rating filter.')
    tmp.replace(out)
    print(f'Wrote {written:,} positions to {out}')


if __name__ == '__main__':
    main()
