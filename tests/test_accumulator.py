import random
import unittest
import chess
import numpy as np
from fastchess.accumulator import Accumulator
from fastchess.features import active_features
from fastchess.model import Evaluator
from fastchess.search import Search


def scratch(weights, board, view):
    return weights['w1'][active_features(board, view)].sum(axis=0) + weights['b1']


class AccumulatorTest(unittest.TestCase):
    """The accumulator is only safe if it never drifts from a from-scratch rebuild,
    so these walk random games and compare after every push and pop."""

    def random_games(self, evaluator, games=25, plies=60, seed=9):
        accumulator = Accumulator(evaluator.weights)
        rng = random.Random(seed)
        worst, specials = 0., {'castle': 0, 'ep': 0, 'promotion': 0}
        for _ in range(games):
            board = chess.Board()
            accumulator.reset(board)
            depth = 0
            for _ in range(plies):
                moves = list(board.legal_moves)
                if not moves:
                    break
                # Bias towards the move types that touch more than two squares.
                odd = [m for m in moves if board.is_castling(m) or board.is_en_passant(m) or m.promotion]
                move = rng.choice(odd) if odd and rng.random() < .6 else rng.choice(moves)
                specials['castle'] += board.is_castling(move)
                specials['ep'] += board.is_en_passant(move)
                specials['promotion'] += bool(move.promotion)
                board.push(move)
                accumulator.push(board)
                depth += 1
                worst = max(worst, abs(accumulator.value(board) - evaluator(board)))
                if evaluator.weights is not None:
                    for index, view in ((0, chess.WHITE), (1, chess.BLACK)):
                        want = scratch(evaluator.weights, board, view)
                        worst = max(worst, float(np.abs(accumulator.acc[index] - want).max()))
                if rng.random() < .25 and depth:      # interleave pops with pushes
                    board.pop()
                    accumulator.pop()
                    depth -= 1
            while depth:                              # unwind and check the way back
                board.pop()
                accumulator.pop()
                depth -= 1
                worst = max(worst, abs(accumulator.value(board) - evaluator(board)))
        return worst, specials

    def test_matches_from_scratch_including_special_moves(self):
        worst, specials = self.random_games(Evaluator('runs/small/best.npz'))
        for name, count in specials.items():
            self.assertGreater(count, 0, f'{name} was never exercised')
        self.assertLess(worst, .01)   # float32 accumulation noise only

    def test_named_special_move_sequences(self):
        # Explicit cases, so coverage of these paths never depends on the random seed.
        evaluator = Evaluator('runs/small/best.npz')
        accumulator = Accumulator(evaluator.weights)
        cases = [
            ('kingside castling', 'r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1', ['e1g1', 'e8c8']),
            ('en passant', '4k3/8/8/8/3p4/8/2P5/4K3 w - - 0 1', ['c2c4', 'd4c3']),
            ('promotion with capture', '1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1', ['a7b8q']),
            ('rook move losing rights', 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', ['a1a5', 'h8h5']),
        ]
        for name, fen, moves in cases:
            with self.subTest(case=name):
                board = chess.Board(fen)
                accumulator.reset(board)
                for uci in moves:
                    board.push_uci(uci)
                    accumulator.push(board)
                    self.assertAlmostEqual(accumulator.value(board), evaluator(board), delta=.01)
                    for index, view in ((0, chess.WHITE), (1, chess.BLACK)):
                        want = scratch(evaluator.weights, board, view)
                        self.assertLess(float(np.abs(accumulator.acc[index] - want).max()), .01)
                for _ in moves:                       # and exactly restored on the way back
                    board.pop()
                    accumulator.pop()
                    self.assertAlmostEqual(accumulator.value(board), evaluator(board), delta=.01)

    def test_material_only_accumulator_is_exact(self):
        worst, _ = self.random_games(Evaluator(), games=15, seed=5)
        self.assertEqual(worst, 0.)

    def test_search_result_is_unchanged(self):
        """Incremental evaluation is an optimisation: same moves, same node counts."""
        evaluator = Evaluator('runs/small/best.npz')

        class Wrapped:            # not an Evaluator, so the accumulator is bypassed
            def __call__(self, board):
                return evaluator(board)

        for fen in ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                    '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1'):
            with self.subTest(fen=fen):
                fast = Search(evaluator).choose(chess.Board(fen), seconds=None, depth=5, nodes=None)
                slow = Search(Wrapped()).choose(chess.Board(fen), seconds=None, depth=5, nodes=None)
                self.assertEqual(fast.move, slow.move)
                self.assertEqual(fast.nodes, slow.nodes)
                self.assertAlmostEqual(fast.score, slow.score, delta=.01)

    def test_reset_clears_state_from_an_earlier_position(self):
        evaluator = Evaluator('runs/small/best.npz')
        accumulator = Accumulator(evaluator.weights)
        board = chess.Board()
        accumulator.reset(board)
        board.push_uci('e2e4')
        accumulator.push(board)
        endgame = chess.Board('8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1')
        accumulator.reset(endgame)
        self.assertFalse(accumulator.stack)
        self.assertAlmostEqual(accumulator.value(endgame), evaluator(endgame), delta=.01)


if __name__ == '__main__':
    unittest.main()
