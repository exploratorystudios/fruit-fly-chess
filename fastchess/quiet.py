"""Reuse existing teacher labels to train a static evaluator on quiet positions.

Search resolves captures and checks before evaluating its leaves. Training on
positions with those tactics still pending teaches the static network to predict
recaptures itself, which can make it discount real material gains in search.
This conservative filter excludes checks, legal captures and imminent promotions.
It is a baseline filter, not a guarantee that a position has no tactical threats.
"""
import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np

from .features import encode
from .positions import read


def is_quiet(board):
    promoting = board.pawns & board.occupied_co[board.turn] & (
        chess.BB_RANK_7 if board.turn else chess.BB_RANK_2)
    return (not board.is_check() and not promoting
            and not any(board.generate_legal_captures())
            and not board.is_game_over())


def select(rows, features, groups):
    """Validate every row before using FENs to filter an already labelled file."""
    if len(rows) != len(features) or len(rows) != len(groups):
        raise ValueError('Positions and labels must have the same number of rows')
    keep = []
    for index, (game, fen) in enumerate(rows):
        board = chess.Board(fen)
        if game != groups[index] or not np.array_equal(encode(board), features[index]):
            raise ValueError(f'Positions and labels disagree at row {index}; use the original positions file')
        if is_quiet(board):
            keep.append(index)
    return np.asarray(keep, dtype=np.int64)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--positions', required=True, help='Original TSV used to create --data')
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        p.error(f'{out} already exists; choose another --out')
    with np.load(a.data, allow_pickle=False) as data:
        x, cp, groups = data['X'], data['cp'], data['game']
    if len(cp) != len(x):
        p.error('Scores and features must have the same number of rows')
    print(f'Checking alignment and filtering {len(x):,} positions...', flush=True)
    try:
        keep = select(read(a.positions), x, groups)
    except ValueError as error:
        p.error(str(error))
    if len(np.unique(groups[keep])) < 2:
        p.error('Need quiet positions from at least two source games')
    metadata = dict(source=str(Path(a.data).resolve()),
                    source_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
                    positions=str(Path(a.positions).resolve()),
                    filter='no check, legal capture, seventh-rank pawn or terminal position',
                    source_positions=len(x), retained=len(keep))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, X=x[keep], cp=cp[keep], game=groups[keep],
                        metadata=json.dumps(metadata))
    tmp.replace(out)
    print(f'Saved {len(keep):,} quiet positions to {out}; source game IDs preserved')


if __name__ == '__main__':
    main()
