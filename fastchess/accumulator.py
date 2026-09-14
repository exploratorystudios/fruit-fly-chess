"""Incrementally maintained first-layer sums.

The network's features are relative to the side to move, so one running sum would
be invalidated every ply. Keeping a separate accumulator per point of view makes
each one a pure function of the position, so a move only ever adds and removes the
handful of features it actually changed. Evaluation then picks the view matching
`board.turn`, and material is tracked the same way because the material term is
simply negated between the two views.

Both views live in one (2, width) array and every feature's two rows are
pre-paired, so updating a feature is a single NumPy operation rather than one per
view. At this width NumPy call overhead dominates arithmetic, so halving the
number of calls matters more than the arithmetic it saves.
"""
import chess
import numpy as np
from .features import VALUES, active_features, feature_id

WHITE_VIEW, BLACK_VIEW = 0, 1


def piece_state(board):
    """Twelve bitboards, ordered white-then-black within each piece type."""
    white, black = board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK]
    pawns, knights, bishops = board.pawns, board.knights, board.bishops
    rooks, queens, kings = board.rooks, board.queens, board.kings
    return (pawns & white, pawns & black, knights & white, knights & black,
            bishops & white, bishops & black, rooks & white, rooks & black,
            queens & white, queens & black, kings & white, kings & black)


def castling_flags(board):
    return (board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK))


def white_material(board):
    white, black = board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK]
    total = 0
    for index, attr in enumerate(('pawns', 'knights', 'bishops', 'rooks', 'queens', 'kings')):
        mask = getattr(board, attr)
        total += VALUES[index] * (chess.popcount(mask & white) - chess.popcount(mask & black))
    return float(total)


def paired_rows(weights):
    """For every feature, the two first-layer rows it contributes to, stacked."""
    w1 = weights['w1']
    pieces = np.empty((768, 2, w1.shape[1]), dtype=w1.dtype)
    for piece in range(1, 7):
        for index, color in enumerate((chess.WHITE, chess.BLACK)):
            for square in range(64):
                slot = ((piece - 1) * 2 + index) * 64 + square
                pieces[slot, WHITE_VIEW] = w1[feature_id(piece, color, square, chess.WHITE)]
                pieces[slot, BLACK_VIEW] = w1[feature_id(piece, color, square, chess.BLACK)]
    # Castling slots are ordered (white kingside, white queenside, black kingside,
    # black queenside); features 768/769 are the viewer's own rights, 770/771 the
    # opponent's.
    rights = np.empty((4, 2, w1.shape[1]), dtype=w1.dtype)
    for slot in range(4):
        color, side = slot < 2, slot % 2
        for index, view in enumerate((chess.WHITE, chess.BLACK)):
            rights[slot, index] = w1[768 + side if color == view else 770 + side]
    # En-passant file features are not mirrored, so both views share a row.
    files = np.empty((8, 2, w1.shape[1]), dtype=w1.dtype)
    for file in range(8):
        files[file, WHITE_VIEW] = files[file, BLACK_VIEW] = w1[772 + file]
    return pieces, rights, files


class Accumulator:
    """Mirrors a board's push/pop so the first layer never rebuilds from scratch."""

    def __init__(self, weights):
        self.weights = weights
        self.enabled = weights is not None
        self.stack = []
        self.acc = None
        if self.enabled:
            self.pieces, self.rights, self.files = paired_rows(weights)
            self.halfmove = weights['w1'][780]

    def reset(self, board):
        self.stack.clear()
        self.state = piece_state(board)
        self.flags = castling_flags(board)
        self.ep = board.ep_square
        self.material = white_material(board)
        if self.enabled:
            w1, bias = self.weights['w1'], self.weights['b1']
            self.acc = np.stack([w1[active_features(board, chess.WHITE)].sum(axis=0) + bias,
                                 w1[active_features(board, chess.BLACK)].sum(axis=0) + bias])

    def add(self, piece, color, square, sign):
        value = VALUES[piece - 1] * sign
        self.material += value if color else -value
        if self.enabled:
            rows = self.pieces[((piece - 1) * 2 + (0 if color else 1)) * 64 + square]
            if sign > 0:
                self.acc += rows
            else:
                self.acc -= rows

    def push(self, board):
        """Call immediately after board.push(); the board is already updated."""
        self.stack.append((self.state, self.flags, self.ep, self.material,
                           self.acc.copy() if self.enabled else None))
        previous, state = self.state, piece_state(board)
        for index in range(12):
            old, new = previous[index], state[index]
            if old == new:
                continue
            piece, color = index // 2 + 1, index % 2 == 0
            for square in chess.scan_forward(old & ~new):
                self.add(piece, color, square, -1)
            for square in chess.scan_forward(new & ~old):
                self.add(piece, color, square, 1)
        self.state = state

        flags = castling_flags(board)
        if flags != self.flags:
            if self.enabled:
                for slot in range(4):
                    was, now = self.flags[slot], flags[slot]
                    if was == now:
                        continue
                    if now:
                        self.acc += self.rights[slot]
                    else:
                        self.acc -= self.rights[slot]
            self.flags = flags

        if board.ep_square != self.ep:
            if self.enabled:
                if self.ep is not None:
                    self.acc -= self.files[self.ep & 7]
                if board.ep_square is not None:
                    self.acc += self.files[board.ep_square & 7]
            self.ep = board.ep_square

    def pop(self):
        """Call immediately after board.pop()."""
        self.state, self.flags, self.ep, self.material, acc = self.stack.pop()
        if acc is not None:
            self.acc = acc

    def value(self, board):
        base = self.material if board.turn else -self.material
        if not self.enabled:
            return base
        weights = self.weights
        hidden = self.acc[WHITE_VIEW if board.turn else BLACK_VIEW] \
            + self.halfmove * (min(board.halfmove_clock, 100) * .01)
        np.maximum(hidden, 0, out=hidden)
        hidden = hidden @ weights['w2'] + weights['b2']
        np.maximum(hidden, 0, out=hidden)
        score = base + (hidden @ weights['w3'] + weights['b3']).item() * 100
        # A builtin clamp; np.clip on a Python float costs more than the layer above.
        return -20000. if score < -20000. else (20000. if score > 20000. else score)
