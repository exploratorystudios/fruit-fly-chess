import unittest
import chess
from fastchess.benchmark import rating_summary
from fastchess.model import Evaluator
from fastchess.search import Search


class BenchmarkTest(unittest.TestCase):
    def test_balanced_score_matches_anchor(self):
        result = rating_summary(dict(win=4, draw=2, loss=4), 1320)
        self.assertEqual(result['points'], 5)
        self.assertEqual(result['performance_elo'], 1320)
        low, high = result['approximate_95pct_interval']
        self.assertLess(low, 1320)
        self.assertGreater(high, 1320)

    def test_zero_score_reports_bound_not_invented_rating(self):
        result = rating_summary(dict(win=0, draw=0, loss=10), 1320)
        self.assertIsNone(result['performance_elo'])
        self.assertLess(result['one_sided_95pct_upper_elo'], 1320)

    def test_untimed_search_completes_depth(self):
        board = chess.Board()
        result = Search(Evaluator()).choose(board, seconds=None, nodes=None, depth=2)
        self.assertEqual(result.depth, 2)
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(board.fen(), chess.STARTING_FEN)

    def test_draw_heavy_interval_is_tighter_than_binomial(self):
        # Four draws carry half a point each and add no spread, so the interval
        # must be narrower than treating the same score as coin flips.
        drawish = rating_summary(dict(win=7, draw=4, loss=1), 0)
        decisive = rating_summary(dict(win=9, draw=0, loss=3), 0)
        self.assertAlmostEqual(drawish['score_fraction'], decisive['score_fraction'])
        span = lambda r: r['approximate_95pct_interval'][1] - r['approximate_95pct_interval'][0]
        self.assertLess(span(drawish), span(decisive))
        # A 9/12 result over a same-strength opponent is a real gain, not noise.
        self.assertGreater(drawish['approximate_95pct_interval'][0], 0)

    def test_identical_results_fall_back_to_wilson(self):
        result = rating_summary(dict(win=0, draw=6, loss=0), 1320)
        low, high = result['approximate_95pct_interval']
        self.assertLess(low, 1320)
        self.assertGreater(high, 1320)

    def test_unbounded_interval_is_explained_not_just_null(self):
        # 8.5/10 pushes the upper bound past a 100% score, where Elo is infinite.
        result = rating_summary(dict(win=8, draw=1, loss=1), 1320)
        low, high = result['approximate_95pct_interval']
        self.assertIsNotNone(low)
        self.assertIsNone(high)
        self.assertIn('unbounded', result['interval_note'])
        self.assertIsNotNone(result['performance_elo'])
