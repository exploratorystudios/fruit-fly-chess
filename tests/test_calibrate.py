import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import chess
import numpy as np

from fastchess.accumulator import Accumulator
from fastchess.calibrate import calibrate
from fastchess.model import Evaluator


class CalibrationTest(unittest.TestCase):
    def test_scales_only_correction_and_accumulator_agrees(self):
        source = Path('runs/teacher500k/best.npz')
        if not source.exists():
            source = Path('runs/small/best.npz')
        original = source.read_bytes()
        full, material = Evaluator(source), Evaluator()
        with tempfile.TemporaryDirectory() as directory:
            for scale in (0., .5, 1.):
                out = Path(directory) / f'{scale}.npz'
                calibrate(source, out, scale)
                evaluator = Evaluator(out)
                accumulator = Accumulator(evaluator.weights)
                board = chess.Board()
                accumulator.reset(board)
                for san in ['e4', 'd5', 'exd5', 'Qxd5', 'Nc3', 'Qd8', 'd4', 'Nf6']:
                    board.push_san(san)
                    accumulator.push(board)
                    expected = material(board) + scale * (full(board) - material(board))
                    self.assertAlmostEqual(evaluator(board), expected, delta=.01)
                    self.assertAlmostEqual(accumulator.value(board), expected, delta=.01)
                for _ in range(len(board.move_stack)):
                    board.pop()
                    accumulator.pop()
                    self.assertAlmostEqual(accumulator.value(board), evaluator(board), delta=.01)
                with np.load(out, allow_pickle=False) as data:
                    meta = json.loads(str(data['calibration']))
                    self.assertEqual(meta['source_sha256'], hashlib.sha256(original).hexdigest())
                with self.assertRaisesRegex(ValueError, 'exists'):
                    calibrate(source, out, scale)
            for invalid in (-1, 2, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    calibrate(source, Path(directory) / 'bad.npz', invalid)
        self.assertEqual(source.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
