import base64
import json
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANATOMY = ROOT / 'public' / 'fly'
CHECKPOINT = ROOT / 'model' / 'fly.pt'


@unittest.skipUnless((ANATOMY / 'anatomy.json').is_file(), 'anatomy not exported')
class AnatomyTest(unittest.TestCase):
    """The whole point of this view is that the positions are real, so check the
    exported asset against its own manifest rather than trusting it."""

    def setUp(self):
        self.meta = json.loads((ANATOMY / 'anatomy.json').read_text())
        self.blob = (ANATOMY / 'anatomy.bin').read_bytes()

    def array(self, name):
        spec = self.meta['arrays'][name]
        dtype = {'float32': np.float32, 'uint16': np.uint16,
                 'uint8': np.uint8, 'int8': np.int8}[spec['dtype']]
        return np.frombuffer(self.blob, dtype=dtype, count=spec['length'], offset=spec['offset'])

    def test_every_neuron_has_a_position(self):
        positions = self.array('positions').reshape(-1, 3)
        self.assertEqual(len(positions), self.meta['neurons'])
        self.assertTrue(np.isfinite(positions).all(), 'a neuron has no coordinate')
        # Centred and scaled, but still the brain's real proportions: not a cube.
        span = positions.max(axis=0) - positions.min(axis=0)
        self.assertAlmostEqual(float(span.max()), 2.0, delta=.01)
        self.assertLess(float(span.min()), float(span.max()),
                        'axes are equal, which would mean the shape was normalised away')

    def test_edges_reference_real_neurons(self):
        for name in ('edgePre', 'edgePost'):
            edge = self.array(name)
            self.assertEqual(len(edge), self.meta['synapsesDrawn'])
            self.assertLess(int(edge.max()), self.meta['neurons'])
        signs = set(np.unique(self.array('edgeSign')).tolist())
        self.assertTrue(signs <= {-1, 1}, f'Dale signs should be +/-1, got {signs}')

    def test_manifest_is_honest_about_what_is_drawn(self):
        self.assertLess(self.meta['synapsesDrawn'], self.meta['synapsesTotal'])
        self.assertEqual(self.meta['neurons'], 6000)
        self.assertIn('FlyWire', self.meta['source'])


@unittest.skipUnless(CHECKPOINT.is_file(), 'model/fly.pt not present')
class FlyEngineTest(unittest.TestCase):
    """The visualiser animates the raster, so the raster must be exactly what the
    network fired — not a summary or a re-simulation."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastchess.fly_engine import FlyEngine
        except ImportError as error:
            raise unittest.SkipTest(f'torch unavailable: {error}')
        cls.engine = FlyEngine(str(CHECKPOINT))

    def test_raster_round_trips_and_matches_the_totals(self):
        import chess
        thought = self.engine.think(chess.Board(), seed=3)
        packed = np.frombuffer(base64.b64decode(thought['raster']), np.uint8)
        frames = np.unpackbits(packed.reshape(thought['timesteps'], -1), axis=1)
        frames = frames[:, :thought['neurons']]
        self.assertEqual(int(frames.sum()), thought['spikesTotal'])
        self.assertEqual(frames.sum(axis=1).tolist(), thought['perTimestep'])
        self.assertEqual(frames.shape, (thought['timesteps'], thought['neurons']))

    def test_plays_a_legal_move_with_scored_alternatives(self):
        import chess
        board = chess.Board()
        thought = self.engine.think(board, seed=5)
        self.assertIn(chess.Move.from_uci(thought['move']), board.legal_moves)
        self.assertTrue(thought['top'])
        scores = [m['score'] for m in thought['top']]
        self.assertEqual(scores, sorted(scores, reverse=True), 'alternatives must be ranked')
        self.assertTrue(all(np.isfinite(scores)), 'illegal moves must be masked out')


if __name__ == '__main__':
    unittest.main()
