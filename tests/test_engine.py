import tempfile
import unittest
from pathlib import Path
import chess
import numpy as np
from fastchess.features import encode, floats
from fastchess.model import Evaluator, export, make_model
from fastchess.search import Search, MATE
from fastchess.train import split_games


class FeaturesTest(unittest.TestCase):
    def test_color_symmetry_and_state(self):
        board = chess.Board()
        board.push_uci('e2e4')
        np.testing.assert_array_equal(encode(board), encode(board.mirror()))
        encoded = encode(board)
        self.assertEqual(encoded[772+4], 1)
        other = board.copy()
        other.ep_square = None
        self.assertFalse(np.array_equal(encoded, encode(other)))
        other = board.copy()
        other.castling_rights = 0
        self.assertFalse(np.array_equal(encoded, encode(other)))
        board.halfmove_clock = 50
        self.assertEqual(floats(encode(board))[-1], .5)

    def test_export_matches_training_forward(self):
        import torch
        torch.set_num_threads(1)
        model = make_model()
        # Nonzero random head checks the whole computation, not just material.
        torch.nn.init.normal_(model.layers[4].weight, std=.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'net.npz'
            export(model, path)
            evaluate = Evaluator(path)
            board = chess.Board()
            rng = np.random.default_rng(8)
            for _ in range(30):
                x = torch.from_numpy(floats(encode(board))).unsqueeze(0)
                self.assertAlmostEqual(model(x).item(), evaluate(board), delta=.001)
                board.push(list(board.legal_moves)[rng.integers(board.legal_moves.count())])

    def test_validation_games_disjoint(self):
        groups = np.repeat(np.arange(20), 5)
        train, val = split_games(groups, 42)
        self.assertFalse(set(groups[train]) & set(groups[val]))
        self.assertEqual(len(train) + len(val), len(groups))


class SearchTest(unittest.TestCase):
    def test_mate_and_restored_board(self):
        board = chess.Board('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1')
        before = board.fen()
        result = Search(Evaluator()).choose(board, seconds=2, depth=2)
        self.assertEqual(board.fen(), before)
        board.push(result.move)
        self.assertTrue(board.is_checkmate())
        self.assertGreater(result.score, MATE-10)

    def test_terminal_positions(self):
        for fen, score in [('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1', -MATE),
                           ('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1', 0)]:
            result = Search(Evaluator()).choose(chess.Board(fen))
            self.assertIsNone(result.move)
            self.assertEqual(result.score, score)

    def test_timeout_preserves_history_and_returns_legal_move(self):
        board = chess.Board()
        board.push_uci('e2e4')
        before, history = board.fen(), list(board.move_stack)
        result = Search(Evaluator()).choose(board, nodes=3)
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(board.fen(), before)
        self.assertEqual(board.move_stack, history)

    def test_underpromotion_to_avoid_stalemate(self):
        # Queen promotion stalemates; rook promotion keeps a winning position.
        board = chess.Board('8/k1P5/2K5/8/8/8/8/8 w - - 0 1')
        result = Search(Evaluator()).choose(board, seconds=2, depth=1)
        self.assertEqual(result.move.uci(), 'c7c8r')

    def test_repetition_is_draw(self):
        board = chess.Board()
        for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2:
            board.push_uci(move)
        result = Search(Evaluator()).choose(board)
        self.assertIsNone(result.move)
        self.assertEqual(result.score, 0)

    def test_wins_hanging_queen(self):
        board = chess.Board('4k3/8/8/8/8/8/q7/R3K3 w Q - 0 1')
        result = Search(Evaluator()).choose(board, seconds=2, depth=2)
        self.assertEqual(result.move.uci(), 'a1a2')


if __name__ == '__main__':
    unittest.main()


class SearchHeuristicsTest(unittest.TestCase):
    """Covers the transposition table, null move, LMR and killer/history ordering.
    Expected moves were confirmed against Stockfish 16 at depth 20."""

    TACTICS = [
        ('6k1/5ppp/8/8/8/8/8/R3K2R w KQ - 0 1', 'a1a8'),      # back-rank mate in one
        ('4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1', 'c3d5'),        # knight forks the queen
        ('r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1', 'f6a6'),    # mate in three
        ('8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1', 'b4f4'),  # wins the f4 pawn
    ]

    def test_finds_tactical_moves(self):
        for fen, expected in self.TACTICS:
            with self.subTest(fen=fen):
                result = Search(Evaluator()).choose(chess.Board(fen), seconds=None,
                                                   depth=6, nodes=None)
                self.assertEqual(result.move.uci(), expected)

    def test_mate_distance_survives_transposition_table(self):
        # Mate in three is five plies, so the reported score must be MATE - 5 exactly.
        # A table that stored ply-relative scores would drift here.
        result = Search(Evaluator()).choose(chess.Board('r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1'),
                                            seconds=None, depth=6, nodes=None)
        self.assertEqual(result.score, MATE - 5)

    def test_null_move_and_reductions_restore_the_board(self):
        # Deep enough to trigger null move (depth >= 3) and LMR (index >= 3).
        board = chess.Board('r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 9')
        before, history = board.fen(), list(board.move_stack)
        Search(Evaluator()).choose(board, seconds=None, depth=5, nodes=None)
        self.assertEqual(board.fen(), before)
        self.assertEqual(board.move_stack, history)

    def test_pruning_cuts_nodes_against_the_frozen_baseline(self):
        from fastchess.search_baseline import Search as Baseline
        fen = 'r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 9'
        old = Baseline(Evaluator()).choose(chess.Board(fen), seconds=None, depth=4, nodes=None)
        new = Search(Evaluator()).choose(chess.Board(fen), seconds=None, depth=4, nodes=None)
        self.assertLess(new.nodes, old.nodes / 3)
        self.assertEqual(new.depth, 4)

    def test_tables_do_not_leak_between_searches(self):
        engine = Search(Evaluator())
        engine.choose(chess.Board(), seconds=None, depth=4, nodes=None)
        self.assertTrue(engine.tt)
        engine.choose(chess.Board('4k3/8/8/8/8/8/8/4K2R w K - 0 1'), seconds=None, depth=2, nodes=None)
        stale = [entry for key, entry in engine.tt.items() if key[0] == chess.Board()._transposition_key()[0]]
        self.assertFalse(stale)
        self.assertTrue(all(slot == [None, None] or slot[0] is not None for slot in engine.killers))


class InspectTest(unittest.TestCase):
    """The visualiser reads Evaluator.inspect(); it must agree with what the
    engine actually plays, or the picture is of a different network."""

    def test_inspect_matches_call(self):
        import random
        for path in ('runs/small/best.npz', 'runs/teacher500k-calibrated/best.npz'):
            evaluator = Evaluator(path)
            board, rng = chess.Board(), random.Random(4)
            for _ in range(40):
                detail = evaluator.inspect(board)
                self.assertAlmostEqual(detail['score'], evaluator(board), delta=1e-6)
                self.assertEqual(len(detail['hidden1']), 128)
                self.assertEqual(len(detail['hidden2']), 32)
                self.assertTrue(all(v >= 0 for v in detail['hidden1']), 'ReLU output must be >= 0')
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(moves[rng.randrange(len(moves))])

    def test_inspect_without_weights_is_material_only(self):
        detail = Evaluator().inspect(chess.Board())
        self.assertEqual(detail['score'], detail['material'])


class ServerlessTest(unittest.TestCase):
    """The deployed site and `run.sh site` must run the same dispatch, or the
    visualisation can show one thing while the deployment does another."""

    def setUp(self):
        from fastchess.site_server import Engine
        from pathlib import Path
        model = Path(__file__).resolve().parents[1] / 'model' / 'best.npz'
        if not model.is_file():
            self.skipTest('model/best.npz not present')
        self.engine = Engine(str(model), seconds=.2, depth=4, max_seconds=1.)

    def test_every_action_responds(self):
        from fastchess.site_server import dispatch
        start = chess.Board().fen()
        for action, request, expect in (
                ('info', {}, 'model'), ('position', {}, 'evaluation'),
                ('candidates', {}, 'candidates'), ('weights', {}, 'shape'),
                ('move', {'fen': start, 'uci': 'e2e4'}, 'fen'),
                ('think', {'fen': start}, 'thought')):
            with self.subTest(action=action):
                status, payload = dispatch(self.engine, action, request)
                self.assertEqual(status, 200)
                self.assertIn(expect, payload)

    def test_bad_input_is_rejected_not_raised(self):
        from fastchess.site_server import dispatch
        for action, request in (('position', {'fen': 'nonsense'}),
                                ('move', {'uci': 'e2e5'}),
                                ('move', {'uci': 'not-a-move'}),
                                ('nope', {})):
            with self.subTest(action=action, request=request):
                status, payload = dispatch(self.engine, action, request)
                self.assertGreaterEqual(status, 400)
                self.assertIn('error', payload)

    def test_think_seconds_are_clamped(self):
        from fastchess.site_server import dispatch
        status, payload = dispatch(self.engine, 'think', {'seconds': 10_000})
        self.assertEqual(status, 200)
        self.assertLess(payload['thought']['seconds'], self.engine.max_seconds + 1)
