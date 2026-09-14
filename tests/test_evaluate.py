import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class EvaluationTest(unittest.TestCase):
    def test_fixed_nodes_and_paired_custom_openings_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            openings = work / 'openings.json'
            openings.write_text(json.dumps([['e2e4', 'e7e5']]))
            out = work / 'match.json'
            command = [sys.executable, '-m', 'fastchess.evaluate',
                       '--model', 'runs/small/best.npz',
                       '--opponent', 'model', '--rival-model', 'runs/small/best.npz',
                       '--games', '2', '--parallel', '1', '--fixed-nodes', '--nodes', '1',
                       '--max-plies', '2', '--openings-file', str(openings), '--out', str(out)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out.read_text())
            self.assertIsNone(report['settings']['seconds'])
            self.assertEqual(report['counts']['unfinished'], 2)
            self.assertEqual(report['model_sha256'], report['rival_model_sha256'])
            for game in report['games']:
                self.assertEqual(game['moves'][:2], ['e2e4', 'e7e5'])
                self.assertEqual(game['mean_nodes'], 1)
                self.assertEqual(game['rival_mean_nodes'], 1)
            self.assertEqual([g['color'] for g in report['games']], ['white', 'black'])
            openings.write_text(json.dumps([['e2e5']]))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Invalid openings file', result.stderr)


if __name__ == '__main__':
    unittest.main()
