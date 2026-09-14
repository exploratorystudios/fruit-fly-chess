"""Frozen pre-optimisation search, kept only as an A/B baseline for benchmarks."""
import time
from dataclasses import dataclass
import chess
from .features import VALUES

MATE = 30000
INF = 32000


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
    def __init__(self, evaluator):
        self.evaluate = evaluator

    def terminal(self, board, ply):
        if board.is_checkmate():
            return -MATE + ply
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_fifty_moves() or board.is_repetition(3)):
            return 0
        return None

    def tick(self):
        self.nodes += 1
        if self.nodes >= self.max_nodes or time.monotonic() >= self.deadline:
            raise Timeout

    def ordered(self, board, moves, preferred=None):
        def rank(move):
            if move == preferred:
                return 100000
            victim = board.piece_type_at(move.to_square)
            capture = (VALUES[victim - 1] if victim else 100 if board.is_en_passant(move) else 0)
            attacker = board.piece_type_at(move.from_square)
            return (10 * capture - (VALUES[attacker - 1] if capture else 0)
                    + (VALUES[move.promotion - 1] if move.promotion else 0))
        return sorted(moves, key=rank, reverse=True)

    def quiescence(self, board, alpha, beta, ply, remaining=8):
        self.tick()
        term = self.terminal(board, ply)
        if term is not None:
            return term
        checked = board.is_check()
        # Hard limit prevents pathological sequences from exhausting Python's stack.
        if ply >= 64:
            return self.evaluate(board)
        if not checked:
            stand = self.evaluate(board)
            if stand >= beta:
                return stand
            alpha = max(alpha, stand)
            if remaining <= 0:
                return alpha
        moves = list(board.legal_moves)
        if not checked:
            moves = [m for m in moves if board.is_capture(m) or m.promotion]
        for move in self.ordered(board, moves):
            board.push(move)
            try:
                value = -self.quiescence(board, -beta, -alpha, ply+1, remaining-1)
            finally:
                board.pop()
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return alpha

    def negamax(self, board, depth, alpha, beta, ply):
        self.tick()
        term = self.terminal(board, ply)
        if term is not None:
            return term
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)
        # Move-order cache only: no cached scores that could ignore draw history.
        key = (board.board_fen(), board.turn, board.castling_rights, board.ep_square)
        best_move, best = None, -INF
        for move in self.ordered(board, board.legal_moves, self.order_cache.get(key)):
            board.push(move)
            try:
                value = -self.negamax(board, depth-1, -beta, -alpha, ply+1)
            finally:
                board.pop()
            if value > best:
                best, best_move = value, move
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        if len(self.order_cache) < 50000:
            self.order_cache[key] = best_move
        return best

    def choose(self, board, seconds=1., depth=6, nodes=1000000):
        if (seconds is not None and seconds <= 0) or depth < 1 or (nodes is not None and nodes < 1):
            raise ValueError('seconds, depth and nodes must be positive')
        start = time.monotonic()
        self.deadline = start + seconds if seconds is not None else float('inf')
        self.max_nodes = nodes if nodes is not None else float('inf')
        self.nodes, self.order_cache = 0, {}
        legal = list(board.legal_moves)
        term = self.terminal(board, 0)
        if term is not None:
            return Result(None, term, 0, 0, time.monotonic()-start)
        best_move, best_score, completed = legal[0], self.evaluate(board), 0
        for d in range(1, depth+1):
            candidate, score = best_move, -INF
            try:
                for move in self.ordered(board, legal, best_move):
                    self.tick()
                    board.push(move)
                    try:
                        value = -self.negamax(board, d-1, -INF, -score, 1)
                    finally:
                        board.pop()
                    if value > score:
                        score, candidate = value, move
                best_move, best_score, completed = candidate, score, d
            except Timeout:
                break
            if abs(best_score) >= MATE - 64:
                break
        return Result(best_move, best_score, completed, self.nodes, time.monotonic()-start)
