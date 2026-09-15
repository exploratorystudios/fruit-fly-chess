"""Iterative deepening negamax with a transposition table, null-move pruning,
late move reductions, killer/history ordering and neural leaf evaluation."""
import time
from dataclasses import dataclass
import chess
from .accumulator import Accumulator
from .features import VALUES
from .model import Evaluator

MATE = 30000
INF = 32000
MAX_PLY = 64
# Scores beyond this magnitude encode a forced mate and need ply correction.
MATE_BOUND = MATE - MAX_PLY
EXACT, LOWER, UPPER = 0, 1, 2
TT_LIMIT = 400000
# Ordering bands, wide enough that history scores can never reach the band above.
TT_BONUS, CAPTURE_BONUS, KILLER_BONUS = 1 << 30, 1 << 20, 1 << 19
# A connectome prior guides root ordering while the evaluator still scores the
# positions reached by search. Keep this below the transposition-table band.
POLICY_BONUS = 1 << 22
# Pruning margins in centipawns per ply of remaining depth.
REVERSE_FUTILITY_DEPTH, REVERSE_FUTILITY_MARGIN = 6, 120
FUTILITY_DEPTH, FUTILITY_MARGIN = 2, 150


class Timeout(Exception):
    pass


@dataclass
class Result:
    move: object
    score: float
    depth: int
    nodes: int
    seconds: float


class Search:
    def __init__(self, evaluator, pruning=True):
        self.evaluate = evaluator
        # Toggle so the futility pruning can be A/B tested against its own absence.
        self.pruning = pruning
        # Only our own Evaluator exposes weights in the layout the accumulator
        # understands; anything else keeps the from-scratch path.
        self.accumulator = Accumulator(evaluator.weights) if isinstance(evaluator, Evaluator) else None
        self.nodes = 0
        self.deadline, self.max_nodes = float('inf'), float('inf')
        self.reset_tables()

    def reset_tables(self):
        # Cleared per search: a table carried between moves could reuse a score
        # that was only correct given the earlier position's repetition history.
        self.tt = {}
        self.killers = [[None, None] for _ in range(MAX_PLY + 1)]
        self.history = {}

    def leaf(self, board):
        return self.evaluate(board) if self.accumulator is None else self.accumulator.value(board)

    def push(self, board, move):
        board.push(move)
        if self.accumulator is not None:
            self.accumulator.push(board)

    def pop(self, board):
        board.pop()
        if self.accumulator is not None:
            self.accumulator.pop()

    def tick(self):
        self.nodes += 1
        if self.nodes >= self.max_nodes:
            raise Timeout
        # Polling the clock every node costs more than the search saves.
        if not self.nodes & 1023 and time.monotonic() >= self.deadline:
            raise Timeout

    def drawn(self, board):
        """Draws that need no move generation. Mate and stalemate are detected
        by the caller, which has the legal move list already."""
        return (board.is_insufficient_material() or board.halfmove_clock >= 100
                or board.is_repetition(3))

    def ordered(self, board, moves, tt_move=None, ply=0, policy=None):
        killer_a, killer_b = self.killers[ply] if ply < len(self.killers) else (None, None)
        history, turn = self.history, board.turn
        scored = []
        for move in moves:
            if move == tt_move:
                scored.append((TT_BONUS, move))
                continue
            victim = board.piece_type_at(move.to_square)
            if victim is not None or board.is_en_passant(move):
                value = VALUES[victim - 1] if victim is not None else 100
                attacker = board.piece_type_at(move.from_square)
                score = CAPTURE_BONUS + 10 * value - VALUES[attacker - 1]
                # A capture that loses material is worse than a decent quiet move,
                # so demote it below the killer band instead of trying it first.
                if value < VALUES[attacker - 1] and self.see(board, move) < 0:
                    score -= CAPTURE_BONUS
            elif move.promotion:
                score = CAPTURE_BONUS
            elif move == killer_a:
                score = KILLER_BONUS + 1
            elif move == killer_b:
                score = KILLER_BONUS
            else:
                score = history.get((turn, move.from_square, move.to_square), 0)
            if move.promotion:
                score += VALUES[move.promotion - 1]
            if policy is not None:
                score += POLICY_BONUS * policy.get(move.uci(), 0)
            scored.append((score, move))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [move for _, move in scored]

    def store_killer(self, move, ply):
        if ply < len(self.killers):
            slot = self.killers[ply]
            if slot[0] != move:
                slot[1], slot[0] = slot[0], move

    def quiescence(self, board, alpha, beta, ply, remaining=8):
        self.tick()
        if self.drawn(board):
            return 0
        in_check = board.is_check()
        if ply >= MAX_PLY:
            return self.leaf(board)
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
        else:
            stand = self.leaf(board)
            if stand >= beta:
                return stand
            if stand > alpha:
                alpha = stand
            if remaining <= 0:
                return alpha
            moves = list(board.generate_legal_captures())
            promoting = (board.pawns & board.occupied_co[board.turn]
                         & (chess.BB_RANK_7 if board.turn else chess.BB_RANK_2))
            if promoting:
                moves += [m for m in board.generate_legal_moves(promoting)
                          if m.promotion and not board.is_capture(m)]
            if not moves:
                # Cheap: any() stops at the first legal move in almost every position.
                return 0 if not any(board.generate_legal_moves()) else alpha
            # Delta pruning: skip captures that cannot drag the score up to alpha.
            margin = alpha - stand - 200
            if margin > 0:
                moves = [m for m in moves if self.gain(board, m) >= margin]
                if not moves:
                    return alpha
        for move in self.ordered(board, moves, None, ply):
            self.push(board, move)
            try:
                value = -self.quiescence(board, -beta, -alpha, ply + 1, remaining - 1)
            finally:
                self.pop(board)
            if value >= beta:
                return value
            if value > alpha:
                alpha = value
        return alpha

    def see(self, board, move):
        """Material left after the capture sequence on the target square.

        Approximate: python-chess cannot recompute sliding attacks against a
        modified occupancy, so x-ray attackers behind a captured piece are not
        revealed. Good enough for ordering, which is the only place it is used —
        a wrong order costs nodes, never correctness.
        """
        square = move.to_square
        victim = board.piece_type_at(square)
        if victim is None and not board.is_en_passant(move):
            return 0
        gains = [VALUES[victim - 1] if victim is not None else 100]
        current = VALUES[board.piece_type_at(move.from_square) - 1]
        occupied = board.occupied & ~chess.BB_SQUARES[move.from_square]
        side = not board.turn
        while True:
            attackers = board.attackers_mask(side, square) & occupied
            if not attackers:
                break
            for piece in range(1, 7):
                subset = attackers & board.pieces_mask(piece, side)
                if subset:
                    break
            else:
                break
            gains.append(current - gains[-1])
            current = VALUES[piece - 1]
            occupied &= ~chess.BB_SQUARES[chess.lsb(subset)]
            side = not side
        for index in range(len(gains) - 2, -1, -1):
            gains[index] = -max(-gains[index], gains[index + 1])
        return gains[0]

    def gain(self, board, move):
        victim = board.piece_type_at(move.to_square)
        value = VALUES[victim - 1] if victim is not None else (100 if board.is_en_passant(move) else 0)
        return value + (VALUES[move.promotion - 1] if move.promotion else 0)

    def negamax(self, board, depth, alpha, beta, ply, can_null=True):
        self.tick()
        in_check = board.is_check()
        if depth <= 0 and not in_check:
            return self.quiescence(board, alpha, beta, ply)
        # Extending checks keeps forcing lines out of the capture-only search,
        # which is where a shallow engine loses material to forks and skewers.
        if in_check and ply < MAX_PLY:
            depth += 1
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0
        if self.drawn(board):
            return 0
        if ply >= MAX_PLY:
            return self.leaf(board)

        key = board._transposition_key()
        entry = self.tt.get(key)
        tt_move = None
        if entry is not None:
            entry_depth, entry_score, entry_flag, tt_move = entry
            # Near the fifty-move horizon a cached score can outlive the history
            # that justified it, so only trust the stored move there.
            if entry_depth >= depth and ply > 0 and board.halfmove_clock < 90:
                score = entry_score
                if score > MATE_BOUND:
                    score -= ply
                elif score < -MATE_BOUND:
                    score += ply
                if (entry_flag == EXACT
                        or (entry_flag == LOWER and score >= beta)
                        or (entry_flag == UPPER and score <= alpha)):
                    return score

        # Reverse futility: if the static score is already so far above beta that
        # a few plies of normal play cannot pull it back, stop here. Restricted to
        # shallow non-PV nodes away from mate scores, where it is safe.
        static = None
        if self.pruning and not in_check and depth <= REVERSE_FUTILITY_DEPTH and abs(beta) < MATE_BOUND:
            static = self.leaf(board)
            if static - REVERSE_FUTILITY_MARGIN * depth >= beta:
                return static

        # Null move: if passing still fails high, the real move surely does.
        # Skipped in check, near mate scores, and in pawn-only endings (zugzwang).
        if (can_null and not in_check and depth >= 3 and abs(beta) < MATE_BOUND
                and board.occupied_co[board.turn] & ~(board.pawns | board.kings)):
            self.push(board, chess.Move.null())
            try:
                value = -self.negamax(board, depth - 3, -beta, -beta + 1, ply + 1, False)
            finally:
                self.pop(board)
            if value >= beta:
                return beta if abs(value) > MATE_BOUND else value

        # Futility: near the horizon a quiet move cannot rescue a position that is
        # already far below alpha, so only captures, promotions and checks are tried.
        futile = False
        if self.pruning and not in_check and depth <= FUTILITY_DEPTH and abs(alpha) < MATE_BOUND:
            if static is None:
                static = self.leaf(board)
            futile = static + FUTILITY_MARGIN * depth <= alpha

        alpha_original = alpha
        best, best_move = -INF, None
        for index, move in enumerate(self.ordered(board, moves, tt_move, ply)):
            quiet = not move.promotion and not board.is_capture(move)
            if futile and quiet and best_move is not None and not board.gives_check(move):
                continue
            reduction = 0
            if quiet and depth >= 3 and index >= 3 and not in_check:
                reduction = min(2 if index >= 6 else 1, depth - 1)
            self.push(board, move)
            try:
                if index == 0:
                    value = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    value = -self.negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
                    if value > alpha and (reduction or value < beta):
                        value = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                self.pop(board)
            if value > best:
                best, best_move = value, move
            if value > alpha:
                alpha = value
            if alpha >= beta:
                if quiet:
                    self.store_killer(move, ply)
                    slot = (board.turn, move.from_square, move.to_square)
                    self.history[slot] = self.history.get(slot, 0) + depth * depth
                break

        flag = LOWER if best >= beta else (EXACT if best > alpha_original else UPPER)
        score = best
        if score > MATE_BOUND:
            score += ply
        elif score < -MATE_BOUND:
            score -= ply
        if len(self.tt) >= TT_LIMIT:
            self.tt.clear()
        self.tt[key] = (depth, score, flag, best_move)
        return best

    def root(self, board, depth, alpha, beta, preferred, policy=None):
        best, best_move = -INF, None
        for index, move in enumerate(self.ordered(
                board, list(board.legal_moves), preferred, 0, policy)):
            self.tick()
            self.push(board, move)
            try:
                if index == 0:
                    value = -self.negamax(board, depth - 1, -beta, -alpha, 1)
                else:
                    value = -self.negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                    if alpha < value < beta:
                        value = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            finally:
                self.pop(board)
            if value > best:
                best, best_move = value, move
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return best, best_move

    def choose(self, board, seconds=1., depth=6, nodes=1000000, root_policy=None):
        if (seconds is not None and seconds <= 0) or depth < 1 or (nodes is not None and nodes < 1):
            raise ValueError('seconds, depth and nodes must be positive')
        start = time.monotonic()
        self.deadline = start + seconds if seconds is not None else float('inf')
        self.max_nodes = nodes if nodes is not None else float('inf')
        self.nodes = 0
        self.reset_tables()
        if self.accumulator is not None:
            self.accumulator.reset(board)
        legal = list(board.legal_moves)
        if not legal:
            return Result(None, -MATE if board.is_check() else 0, 0, 0, time.monotonic() - start)
        if self.drawn(board):
            return Result(None, 0, 0, 0, time.monotonic() - start)
        best_move, best_score, completed = legal[0], self.leaf(board), 0
        for current in range(1, depth + 1):
            try:
                if current <= 2 or abs(best_score) >= MATE_BOUND:
                    score, move = self.root(board, current, -INF, INF,
                                            best_move if completed else None,
                                            root_policy)
                else:
                    # Aspiration window: re-search wider only when it fails.
                    window = 50
                    while True:
                        low, high = best_score - window, best_score + window
                        score, move = self.root(board, current, low, high,
                                                best_move if completed else None,
                                                root_policy)
                        if low < score < high:
                            break
                        window *= 4
                        if window > 2000:
                            score, move = self.root(board, current, -INF, INF,
                                                    best_move if completed else None,
                                                    root_policy)
                            break
                if move is not None:
                    best_move, best_score, completed = move, score, current
            except Timeout:
                break
            if abs(best_score) >= MATE_BOUND:
                break
        return Result(best_move, best_score, completed, self.nodes, time.monotonic() - start)
