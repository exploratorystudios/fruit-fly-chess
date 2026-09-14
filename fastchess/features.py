import chess
import numpy as np

DIM = 781
VALUES = (100, 320, 330, 500, 900, 0)
MATERIAL = np.zeros(DIM, dtype=np.float32)
for p, value in enumerate(VALUES):
    MATERIAL[p * 128:p * 128 + 64] = value
    MATERIAL[p * 128 + 64:(p + 1) * 128] = -value


def feature_id(piece, color, square, view):
    """Index of one piece-on-square feature from `view`'s point of view."""
    return ((piece - 1) * 2 + (color != view)) * 64 + (square ^ (0 if view else 56))


def active_features(board, view=None):
    """Own pieces face north; rights and en passant follow the same orientation.

    `view` defaults to the side to move. Passing it explicitly gives the same
    encoding from a chosen point of view, which is what lets the accumulator keep
    one running sum per view instead of rebuilding after every ply."""
    if view is None:
        view = board.turn
    flip = 0 if view else 56
    ids = []
    for piece in range(1, 7):
        for opponent in (False, True):
            color = not view if opponent else view
            base = ((piece - 1) * 2 + int(opponent)) * 64
            ids.extend(base + (sq ^ flip) for sq in chess.scan_forward(board.pieces_mask(piece, color)))
    for i, color in enumerate((view, not view)):
        if board.has_kingside_castling_rights(color):
            ids.append(768 + 2 * i)
        if board.has_queenside_castling_rights(color):
            ids.append(769 + 2 * i)
    if board.ep_square is not None:
        ids.append(772 + chess.square_file(board.ep_square))
    return ids


def encode(board):
    x = np.zeros(DIM, dtype=np.uint8)
    x[active_features(board)] = 1
    x[780] = min(board.halfmove_clock, 100)
    return x


def floats(x):
    x = x.astype(np.float32)
    x[..., 780] /= 100
    return x
