import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import chess
import numpy as np
from fastchess.positions import dedup_key, read, sample

PGN = Path(__file__).resolve().parents[2] / 'games' / 'LumbrasGigaBase_OTB_2020-2024.pgn'


@unittest.skipUnless(PGN.exists(), f'{PGN} not available')
class PositionsTest(unittest.TestCase):
    def test_sampling_is_deduplicated_and_deterministic(self):
        first = list(sample(PGN, 120, per_game=8))
        second = list(sample(PGN, 120, per_game=8))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 120)
        keys = [dedup_key(chess.Board(fen)) for _, fen in first]
        self.assertEqual(len(set(keys)), len(keys))

    def test_skips_openings_and_terminal_positions(self):
        for _, fen in sample(PGN, 60, per_game=8):
            board = chess.Board(fen)
            self.assertTrue(board.is_valid())
            self.assertFalse(board.is_game_over())
            self.assertGreaterEqual(board.fullmove_number, 5)

    def test_seeds_give_different_samples(self):
        a = [fen for _, fen in sample(PGN, 60, per_game=8, seed=1)]
        b = [fen for _, fen in sample(PGN, 60, per_game=8, seed=2)]
        self.assertNotEqual(a, b)

    def test_positions_span_several_games(self):
        games = {game for game, _ in sample(PGN, 100, per_game=8)}
        self.assertGreater(len(games), 1, 'training needs positions from multiple games to split')


@unittest.skipUnless(PGN.exists(), f'{PGN} not available')
class LabelTest(unittest.TestCase):
    """Runs the real CLIs, since the point of the two-stage split is that the
    sampled file round-trips into the same .npz layout the trainer already reads."""

    def run_module(self, module, *args):
        result = subprocess.run([sys.executable, '-m', f'fastchess.{module}', *args],
                                capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_two_stage_pipeline_matches_trainer_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            self.run_module('positions', '--pgn', str(PGN), '--positions', '40',
                            '--per-game', '8', '--out', str(work / 'p.tsv'))
            rows = read(work / 'p.tsv')
            self.assertEqual(len(rows), 40)
            self.run_module('label', '--positions', str(work / 'p.tsv'),
                            '--out', str(work / 't.npz'), '--nodes', '800', '--workers', '2')
            with np.load(work / 't.npz', allow_pickle=False) as data:
                self.assertEqual(sorted(data.files), ['X', 'cp', 'game', 'metadata'])
                self.assertEqual(data['X'].shape, (40, 781))
                self.assertEqual(data['X'].dtype, np.uint8)
                self.assertEqual(data['cp'].dtype, np.int16)
                self.assertEqual(data['game'].dtype, np.int32)
                # Row order must still line up with the sampled positions.
                self.assertEqual(list(data['game']), [game for game, _ in rows])
                self.assertEqual(json.loads(str(data['metadata']))['missing'], 0)

    def test_labelling_resumes_from_existing_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            self.run_module('positions', '--pgn', str(PGN), '--positions', '20',
                            '--per-game', '8', '--out', str(work / 'p.tsv'))
            common = ['--positions', str(work / 'p.tsv'), '--out', str(work / 't.npz'),
                      '--nodes', '800', '--workers', '2', '--shards', str(work / 'shards')]
            self.run_module('label', *common)
            before = sorted(path.read_text() for path in (work / 'shards').glob('shard-*.txt'))
            output = self.run_module('label', *common)
            after = sorted(path.read_text() for path in (work / 'shards').glob('shard-*.txt'))
            # A second run relabels nothing and leaves the shards byte-identical.
            self.assertEqual(before, after)
            self.assertIn('Saved 20 labels', output)


if __name__ == '__main__':
    unittest.main()
