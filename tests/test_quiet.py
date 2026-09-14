import unittest

import chess
import numpy as np

from fastchess.features import encode
from fastchess.quiet import is_quiet, select


class QuietTest(unittest.TestCase):
    def test_excludes_forcing_moves_for_both_colors(self):
        cases = [
            ('check', '4k3/8/8/8/8/8/4r3/4K3 w - - 0 1'),
            ('capture', '4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1'),
            ('en passant', '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1'),
            ('promotion', '4k3/P7/8/8/8/8/8/4K3 w - - 0 1'),
            ('terminal', '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1'),
        ]
        for name, fen in cases:
            board = chess.Board(fen)
            for view in (board, board.mirror()):
                with self.subTest(case=name, turn=view.turn):
                    self.assertFalse(is_quiet(view))
        self.assertTrue(is_quiet(chess.Board()))

    def test_filter_preserves_original_row_indices_and_game_ids(self):
        boards = [chess.Board(), chess.Board('4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1')]
        third = chess.Board()
        third.push_uci('e2e4')
        boards.append(third)
        rows = [(game, board.fen(en_passant='fen')) for game, board in zip([11, 12, 13], boards)]
        features = np.stack([encode(board) for board in boards])
        groups = np.array([11, 12, 13])
        kept = select(rows, features, groups)
        np.testing.assert_array_equal(kept, [0, 2])
        np.testing.assert_array_equal(groups[kept], [11, 13])
        with self.assertRaisesRegex(ValueError, 'same number'):
            select(rows[:-1], features, groups)
        with self.assertRaisesRegex(ValueError, 'row 0'):
            select(rows, features[::-1], groups)
        with self.assertRaisesRegex(ValueError, 'row 0'):
            select(rows, features, groups[::-1])


if __name__ == '__main__':
    unittest.main()
