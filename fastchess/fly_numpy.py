"""The fly connectome's forward pass in NumPy, with no PyTorch.

The dynamics are only sparse accumulation and elementwise arithmetic, so the
network does not need a deep-learning runtime to run — which is what lets the
deployed site fire the real neurons instead of showing a still diagram. Outputs
are checked against the PyTorch implementation in the tests.
"""
import base64
import time
from pathlib import Path
import chess
import numpy as np

# Shiu et al. LIF parameters, in mV / ms — identical to the trained model's.
TAU_SYN, TAU_MEM = 5.0, 20.0
V_REST, V_RESET, V_THRESH = -52.0, -52.0, -45.0
T_REFRAC, DT = 2.2, 0.5


class FlyNumpy:
    def __init__(self, path, temperature=.15, repetition_penalty=4.):
        data = np.load(path, allow_pickle=False)
        self.edge_post = data['edge_post'].astype(np.int64)
        self.edge_pre = data['edge_pre'].astype(np.int64)
        self.edge_value = data['edge_value'].astype(np.float32)
        self.sensory = data['sensory_idx'].astype(np.int64)
        self.motor = data['motor_idx'].astype(np.int64)
        self.win_w, self.win_b = data['w_in_weight'], data['w_in_bias']
        self.out_w, self.out_b = data['readout_weight'], data['readout_bias']
        self.neurons = int(data['n_neurons'])
        self.timesteps = int(data['timesteps'])
        self.w_scale = float(data['w_scale'])
        self.name = str(data['name'])
        self.val_top1, self.val_top5 = float(data['val_top1']), float(data['val_top5'])
        self.temperature, self.repetition_penalty = temperature, repetition_penalty
        self.refrac_steps = max(1, int(round(T_REFRAC / DT)))
        self.syn_decay = float(np.exp(-DT / TAU_SYN))
        self.mem_factor = DT / TAU_MEM

    def simulate(self, features):
        """Run the connectome and return (spike raster, motor firing rate)."""
        n = self.neurons
        stim = np.zeros(n, dtype=np.float32)
        np.add.at(stim, self.sensory, self.win_w @ features.astype(np.float32) + self.win_b)

        v = np.full(n, V_REST, dtype=np.float32)
        g = np.zeros(n, dtype=np.float32)
        spikes = np.zeros(n, dtype=np.float32)
        refrac = np.full(n, float(self.refrac_steps), dtype=np.float32)
        raster = np.zeros((self.timesteps, n), dtype=bool)
        motor_trace = np.zeros(len(self.motor), dtype=np.float32)

        for step in range(self.timesteps):
            # One sparse matrix-vector product, written as a weighted bincount.
            rec = self.w_scale * np.bincount(
                self.edge_post, weights=self.edge_value * spikes[self.edge_pre],
                minlength=n).astype(np.float32)
            refrac = np.where(spikes > 0, 0., refrac + 1.)
            gate = (refrac >= self.refrac_steps).astype(np.float32)
            g = g * self.syn_decay + rec * gate
            v = v + stim
            v = v + self.mem_factor * (g - (v - V_REST))
            spikes = (v - V_THRESH > 0).astype(np.float32)
            v = v - (v - V_RESET) * spikes
            g = g - g * spikes
            motor_trace += spikes[self.motor]
            raster[step] = spikes > 0
        return raster, motor_trace / self.timesteps

    def think(self, board, seed=None):
        from .fly.encoding import encode_board, encode_move, legal_mask
        started = time.monotonic()
        raster, rate = self.simulate(encode_board(board))
        logits = (self.out_w @ rate + self.out_b).astype(np.float64)

        scores = np.where(legal_mask(board), logits, -np.inf)
        for move in board.legal_moves:
            board.push(move)
            repeated = board.is_repetition(2)
            board.pop()
            if repeated and self.repetition_penalty:
                scores[encode_move(board, move)] -= self.repetition_penalty

        legal = list(board.legal_moves)
        ranked = sorted(legal, key=lambda m: scores[encode_move(board, m)], reverse=True)
        shifted = scores - scores[np.isfinite(scores)].max()
        probabilities = np.where(np.isfinite(scores),
                                 np.exp(shifted / max(self.temperature, 1e-6)), 0.)
        total = probabilities.sum()
        if total > 0:
            probabilities = probabilities / total
            chosen = np.random.default_rng(seed).choice(len(scores), p=probabilities)
            move = next((m for m in legal if encode_move(board, m) == chosen), ranked[0])
        else:
            move = ranked[0]

        counts = raster.sum(axis=0).astype(np.uint16)
        return {
            'move': move.uci(), 'san': board.san(move),
            'seconds': time.monotonic() - started,
            'timesteps': int(raster.shape[0]), 'neurons': int(self.neurons),
            'spikesTotal': int(raster.sum()), 'meanRate': float(raster.mean()),
            'perTimestep': raster.sum(axis=1).astype(int).tolist(),
            'raster': base64.b64encode(np.packbits(raster, axis=1).tobytes()).decode(),
            'spikeCounts': base64.b64encode(counts.tobytes()).decode(),
            'top': [{'uci': m.uci(), 'san': board.san(m),
                     'score': float(scores[encode_move(board, m)]),
                     'probability': float(probabilities[encode_move(board, m)])}
                    for m in ranked[:8]],
        }

    def describe(self):
        return {'model': self.name, 'kind': 'fly', 'neurons': self.neurons,
                'timesteps': self.timesteps, 'synapses': int(len(self.edge_value)),
                'valTop1': self.val_top1, 'valTop5': self.val_top5,
                'temperature': self.temperature, 'runtime': 'numpy'}
