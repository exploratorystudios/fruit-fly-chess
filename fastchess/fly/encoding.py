"""Turn LumbrasGigaBase PGN into (position, move) training tensors.

The database is annotated broadcast PGN: comments, clock times, engine
variations, and a fair number of games whose movetext is corrupted (source
tags spliced into comments, move numbers out of order). python-chess raises on
those, so every game is parsed defensively and bad ones are dropped.

Encoding
--------
Position -> 773 floats: 12 piece planes x 64 squares, plus side to move,
4 castling rights, and en-passant file presence.
Move     -> a class index in [0, 4095): from_square * 64 + to_square.
Promotions collapse onto the same from/to class; underpromotion is rare enough
that the readout treats it as queen promotion downstream.
"""

import argparse
from pathlib import Path
import numpy as np
import chess
import chess.pgn

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]
FEATURE_DIM = 773
N_MOVE_CLASSES = 64 * 64


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a position from the side-to-move's point of view.

    The board is mirrored when Black is to move, so the network only ever has
    to learn one side's worth of chess.
    """
    if board.turn == chess.BLACK:
        board = board.mirror()

    x = np.zeros(FEATURE_DIM, dtype=np.float32)
    for plane, piece_type in enumerate(PIECE_ORDER):
        for colour_offset, colour in enumerate((chess.WHITE, chess.BLACK)):
            base = (plane * 2 + colour_offset) * 64
            for sq in board.pieces(piece_type, colour):
                x[base + sq] = 1.0

    x[768] = 1.0  # side to move is always "white" after mirroring
    x[769] = float(board.has_kingside_castling_rights(chess.WHITE))
    x[770] = float(board.has_queenside_castling_rights(chess.WHITE))
    x[771] = float(board.has_kingside_castling_rights(chess.BLACK))
    x[772] = float(board.has_queenside_castling_rights(chess.BLACK))
    return x


def encode_move(board: chess.Board, move: chess.Move) -> int:
    """Move index, mirrored to match encode_board's orientation."""
    frm, to = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        frm, to = chess.square_mirror(frm), chess.square_mirror(to)
    return frm * 64 + to


def legal_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask over the 4096 move classes, in mirrored orientation."""
    mask = np.zeros(N_MOVE_CLASSES, dtype=bool)
    for mv in board.legal_moves:
        mask[encode_move(board, mv)] = True
    return mask


def iter_positions(pgn_path, max_games=None, min_elo=0, skip_opening=6,
                   stride=1):
    """Yield (features, move_index) pairs from the PGN.

    skip_opening drops the first N plies of each game: they are near-identical
    across the database and would otherwise dominate the training set.
    """
    kept = games = 0
    with open(pgn_path, 'r', encoding='utf-8', errors='replace') as fh:
        while True:
            if max_games is not None and games >= max_games:
                return
            try:
                game = chess.pgn.read_game(fh)
            except Exception:
                continue
            if game is None:
                return
            games += 1

            if min_elo:
                try:
                    elos = [int(game.headers.get(k, 0)) for k in ('WhiteElo', 'BlackElo')]
                except ValueError:
                    continue
                if min(elos) < min_elo:
                    continue

            board = game.board()
            ply = 0
            try:
                moves = list(game.mainline_moves())
            except Exception:
                continue
            for mv in moves:
                if not board.is_legal(mv):
                    break  # corrupted movetext; keep what we parsed so far
                if ply >= skip_opening and (ply % stride == 0):
                    yield encode_board(board), encode_move(board, mv)
                    kept += 1
                board.push(mv)
                ply += 1


def build_dataset(pgn_path, out_path, n_positions, **kw):
    X = np.zeros((n_positions, FEATURE_DIM), dtype=np.uint8)
    y = np.zeros(n_positions, dtype=np.int16)
    i = 0
    for feat, mv in iter_positions(pgn_path, **kw):
        X[i] = feat
        y[i] = mv
        i += 1
        if i % 50000 == 0:
            print(f'  {i}/{n_positions} positions', flush=True)
        if i >= n_positions:
            break
    X, y = X[:i], y[:i]
    np.savez_compressed(out_path, X=X, y=y)
    print(f'wrote {i} positions -> {out_path}')
    return i


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pgn', default='/home/thewindmage/Documents/chess-ccg/games/LumbrasGigaBase_OTB_2020-2024.pgn')
    p.add_argument('--out', default='data/positions.npz')
    p.add_argument('--n', type=int, default=400000)
    p.add_argument('--min-elo', type=int, default=2200)
    p.add_argument('--stride', type=int, default=3)
    p.add_argument('--max-games', type=int, default=None)
    a = p.parse_args()
    build_dataset(a.pgn, a.out, a.n, min_elo=a.min_elo, stride=a.stride,
                  max_games=a.max_games)


# ---------------------------------------------------------------------------
# Finisher slice: learning how games actually end
# ---------------------------------------------------------------------------
# Only 0.02% of positions in the 2200+ slice have a mate available, because
# strong players resign rather than get mated -- at 2600+ literally none of the
# sampled games reached checkmate. A policy trained on that data has never seen
# a game finish and reliably fails to convert winning material.
#
# This slice fixes the distribution without any play-time machinery: take games
# that ended in checkmate (from ANY rating band, since that is where finishes
# live), keep only their closing plies, and keep only the *winner's* moves --
# the loser's final moves are exactly what we do not want imitated.

def iter_finisher_positions(pgn_path, max_games=None, last_plies=15,
                            min_elo=0):
    """Yield (features, move) from the closing plies of games ending in mate."""
    games = 0
    with open(pgn_path, 'r', encoding='utf-8', errors='replace') as fh:
        while True:
            if max_games is not None and games >= max_games:
                return
            try:
                game = chess.pgn.read_game(fh)
            except Exception:
                continue
            if game is None:
                return
            games += 1

            if min_elo:
                try:
                    if min(int(game.headers.get(k, 0))
                           for k in ('WhiteElo', 'BlackElo')) < min_elo:
                        continue
                except ValueError:
                    continue

            try:
                moves = list(game.mainline_moves())
            except Exception:
                continue
            if not moves:
                continue

            board = game.board()
            legal = []
            for mv in moves:
                if not board.is_legal(mv):
                    break
                legal.append(mv)
                board.push(mv)

            if not board.is_checkmate():
                continue  # resigned, drawn, or truncated -- no finish to learn

            # board.turn is the side that just got mated, so the winner is the
            # other one. Replay and emit only the winner's closing moves.
            winner = not board.turn
            start = max(0, len(legal) - last_plies)
            board = game.board()
            for i, mv in enumerate(legal):
                if i >= start and board.turn == winner:
                    yield encode_board(board), encode_move(board, mv)
                board.push(mv)


def build_mixed_dataset(pgn_path, out_path, main_npz=None, n_main=400000,
                        n_finish=35000, last_plies=15, min_elo=2200, stride=3,
                        max_finish_games=None):
    """Main slice (strong players) + finisher slice (games that actually end)."""
    if main_npz and Path(main_npz).exists():
        d = np.load(main_npz)
        Xm, ym = d['X'][:n_main], d['y'][:n_main]
        print(f'reusing {len(Xm)} main positions from {main_npz}')
    else:
        Xm = np.zeros((n_main, FEATURE_DIM), dtype=np.uint8)
        ym = np.zeros(n_main, dtype=np.int16)
        i = 0
        for feat, mv in iter_positions(pgn_path, min_elo=min_elo, stride=stride):
            Xm[i], ym[i] = feat, mv
            i += 1
            if i % 50000 == 0:
                print(f'  main {i}/{n_main}', flush=True)
            if i >= n_main:
                break
        Xm, ym = Xm[:i], ym[:i]

    Xf = np.zeros((n_finish, FEATURE_DIM), dtype=np.uint8)
    yf = np.zeros(n_finish, dtype=np.int16)
    j = 0
    for feat, mv in iter_finisher_positions(pgn_path, max_games=max_finish_games,
                                            last_plies=last_plies):
        Xf[j], yf[j] = feat, mv
        j += 1
        if j % 5000 == 0:
            print(f'  finisher {j}/{n_finish}', flush=True)
        if j >= n_finish:
            break
    Xf, yf = Xf[:j], yf[:j]

    X = np.concatenate([Xm, Xf]); y = np.concatenate([ym, yf])
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    np.savez_compressed(out_path, X=X, y=y)
    print(f'wrote {len(X)} positions ({len(Xm)} main + {len(Xf)} finisher, '
          f'{len(Xf)/len(X):.1%} finisher) -> {out_path}')
    return len(X)


# ---------------------------------------------------------------------------
# Lichess puzzle slice: high-rated mates, in quantity
# ---------------------------------------------------------------------------
# OTB games cannot supply high-rated mates -- strong players resign, so the
# mate rate falls from 18% at 1000 Elo to 0% at 2600+. The Lichess puzzle
# database solves this directly: millions of positions mined from real games,
# each tagged with themes (mateIn1, mateIn2, ...) and individually rated, with
# plenty of mates rated well above 2500.
#
# CSV columns: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,
#              Themes,GameUrl,OpeningTags
# The FEN is the position *before* the loser's blunder. Moves[0] is that
# blunder; the solver is to move after it, and Moves[1] is the correct reply.
# So each puzzle yields a (position, best-move) pair in exactly our format --
# and for a mateIn1 puzzle, that move delivers mate.

PUZZLE_MATE_THEMES = ('mateIn1', 'mateIn2', 'mateIn3', 'mateIn4', 'mate')


def iter_puzzle_positions(csv_path, themes=PUZZLE_MATE_THEMES, min_rating=0,
                          max_rating=10000, all_solution_plies=True,
                          limit=None):
    """Yield (features, move) pairs from Lichess puzzles matching `themes`.

    all_solution_plies keeps every move of the solution line (so a mateIn2
    teaches the setup move as well as the mate), not just the first.
    """
    import csv as _csv
    import io
    import subprocess as _sp

    want = set(themes) if themes else None
    n = 0

    if str(csv_path).endswith('.zst'):
        proc = _sp.Popen(['zstdcat', str(csv_path)], stdout=_sp.PIPE)
        stream = io.TextIOWrapper(proc.stdout, encoding='utf-8', errors='replace')
    else:
        proc = None
        stream = open(csv_path, 'r', encoding='utf-8', errors='replace')

    try:
        reader = _csv.reader(stream)
        header = next(reader, None)
        for row in reader:
            if limit is not None and n >= limit:
                return
            if len(row) < 8:
                continue
            fen, moves_s, rating_s, themes_s = row[1], row[2], row[3], row[7]
            try:
                rating = int(rating_s)
            except ValueError:
                continue
            if not (min_rating <= rating <= max_rating):
                continue
            if want and not (want & set(themes_s.split())):
                continue

            try:
                board = chess.Board(fen)
                uci = moves_s.split()
                if len(uci) < 2:
                    continue
                board.push(chess.Move.from_uci(uci[0]))   # the blunder
            except Exception:
                continue

            # Remaining plies alternate solver / opponent; emit solver moves.
            for i, u in enumerate(uci[1:]):
                try:
                    mv = chess.Move.from_uci(u)
                    if not board.is_legal(mv):
                        break
                except Exception:
                    break
                if i % 2 == 0:
                    yield encode_board(board), encode_move(board, mv)
                    n += 1
                    if not all_solution_plies:
                        board = None
                        break
                board.push(mv)
    finally:
        stream.close()
        if proc is not None:
            proc.stdout.close()
            proc.wait()
