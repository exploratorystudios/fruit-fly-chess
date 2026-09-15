"""Runs the fly-connectome model and reports which neurons actually fired.

Unlike the small evaluator, this network chooses moves directly: 773 board
features drive 1,024 sensory neurons, the 6,000-neuron FlyWire sub-connectome runs
for 32 timesteps of leaky integrate-and-fire dynamics, and the 1,024 highest
in-degree neurons are read out as a score for every from/to square pair.
"""
import base64
import time
from pathlib import Path
import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


class FlyEngine:
    def __init__(self, checkpoint, device='cpu', temperature=.15, repetition_penalty=4.):
        import torch
        from .fly.brain import FlyBrainChess
        self.torch = torch
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        weights = state['state']
        post, pre = weights['edge_index'].numpy()
        # The checkpoint carries every buffer, so the constructor only needs a
        # stub of the right shape; load_state_dict then restores the real values.
        stub = {'n_neurons': int(max(post.max(), pre.max())) + 1, 'pre': pre, 'post': post,
                'weight': (weights['edge_sign'] * weights['edge_mag']).numpy()}
        self.model = FlyBrainChess(stub, timesteps=int(state.get('timesteps', 32)))
        self.model.load_state_dict(weights)
        self.model.eval().to(device)
        self.device = device
        self.temperature, self.repetition_penalty = temperature, repetition_penalty
        self.neurons = self.model.n_neurons
        self.timesteps = self.model.timesteps
        self.name = Path(checkpoint).stem
        self.val_top1, self.val_top5 = state.get('val_acc'), state.get('val_top5')

    def think(self, board, seed=None):
        from .fly.encoding import encode_board, encode_move, legal_mask
        torch = self.torch
        started = time.monotonic()
        raster = []
        with torch.no_grad():
            x = torch.from_numpy(encode_board(board)).float().unsqueeze(0).to(self.device)
            logits = self.model(x, raster=raster)[0].cpu().numpy().astype(np.float64)

        mask = legal_mask(board)
        scores = np.where(mask, logits, -np.inf)
        for move in board.legal_moves:
            board.push(move)
            repeated = board.is_repetition(2)
            board.pop()
            if repeated and self.repetition_penalty:
                scores[encode_move(board, move)] -= self.repetition_penalty

        legal = list(board.legal_moves)
        ranked = sorted(legal, key=lambda m: scores[encode_move(board, m)], reverse=True)
        finite = scores[np.isfinite(scores)]
        shifted = scores - finite.max()
        probabilities = np.where(np.isfinite(scores), np.exp(shifted / max(self.temperature, 1e-6)), 0.)
        total = probabilities.sum()
        if total > 0:
            probabilities = probabilities / total
            rng = np.random.default_rng(seed)
            chosen = rng.choice(len(scores), p=probabilities)
            move = next((m for m in legal if encode_move(board, m) == chosen), ranked[0])
        else:
            move = ranked[0]

        # 6,000 neurons x 32 timesteps packed to bits: 24 KiB instead of 768 KiB.
        frames = np.stack([r.numpy() for r in raster]) if raster else np.zeros((0, self.neurons), bool)
        counts = frames.sum(axis=0).astype(np.uint16)
        return {
            'move': move.uci(), 'san': board.san(move),
            'seconds': time.monotonic() - started,
            'timesteps': int(frames.shape[0]), 'neurons': int(self.neurons),
            'spikesTotal': int(frames.sum()),
            'meanRate': float(frames.mean()) if frames.size else 0.,
            'perTimestep': frames.sum(axis=1).astype(int).tolist(),
            'raster': base64.b64encode(np.packbits(frames, axis=1).tobytes()).decode(),
            'spikeCounts': base64.b64encode(counts.tobytes()).decode(),
            'top': [{'uci': m.uci(), 'san': board.san(m),
                     'score': float(scores[encode_move(board, m)]),
                     'probability': float(probabilities[encode_move(board, m)])}
                    for m in ranked[:8]],
            'policy': [{'uci': m.uci(),
                        'score': float(scores[encode_move(board, m)])}
                       for m in ranked],
        }

    def describe(self):
        return {'model': self.name, 'kind': 'fly', 'neurons': self.neurons,
                'timesteps': self.timesteps, 'synapses': int(self.model.n_synapses),
                'valTop1': self.val_top1, 'valTop5': self.val_top5,
                'temperature': self.temperature}
