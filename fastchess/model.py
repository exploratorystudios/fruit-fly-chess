import numpy as np
from .features import MATERIAL, active_features

DEFAULT_MODEL = 'runs/teacher500k-calibrated/best.npz'


class Evaluator:
    """Sparse first layer, NumPy inference; PyTorch and Stockfish aren't needed to play."""
    def __init__(self, path=None):
        self.weights = None
        if path:
            with np.load(path, allow_pickle=False) as data:
                self.weights = {k: data[k] for k in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3')}

    def __call__(self, board):
        ids = active_features(board)
        base = float(MATERIAL[ids].sum())
        if self.weights is None:
            return base
        w = self.weights
        h = np.maximum(0, w['w1'][ids].sum(axis=0) + w['b1']
                       + w['w1'][780] * min(board.halfmove_clock, 100) / 100)
        h = np.maximum(0, h @ w['w2'] + w['b2'])
        return float(np.clip(base + (h @ w['w3'] + w['b3']).item() * 100, -20000, 20000))


    def inspect(self, board):
        """Same computation as __call__, but returning every intermediate value.

        Kept separate so the search's hot path stays untouched; a test asserts the
        two agree, so the visualiser can never drift from what the engine plays.
        """
        ids = active_features(board)
        base = float(MATERIAL[ids].sum())
        result = {'features': ids, 'material': base, 'halfmove': min(board.halfmove_clock, 100)}
        if self.weights is None:
            return {**result, 'hidden1': [], 'hidden2': [], 'correction': 0., 'score': base}
        w = self.weights
        pre1 = (w['w1'][ids].sum(axis=0) + w['b1']
                + w['w1'][780] * min(board.halfmove_clock, 100) / 100)
        h1 = np.maximum(0, pre1)
        pre2 = h1 @ w['w2'] + w['b2']
        h2 = np.maximum(0, pre2)
        correction = float((h2 @ w['w3'] + w['b3']).item())
        return {**result,
                'pre1': pre1.tolist(), 'hidden1': h1.tolist(),
                'pre2': pre2.tolist(), 'hidden2': h2.tolist(),
                'correction': correction,
                'score': float(np.clip(base + correction * 100, -20000, 20000))}


def make_model(width=128):
    import torch
    from torch import nn
    from .features import DIM

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(DIM, width), nn.ReLU(),
                                        nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1))
            self.register_buffer('material', torch.from_numpy(MATERIAL.copy()))
            nn.init.zeros_(self.layers[4].weight)
            nn.init.zeros_(self.layers[4].bias)

        def forward(self, x):
            return x @ self.material + 100 * self.layers(x).squeeze(-1)

    return Network()


def export(model, path):
    from pathlib import Path
    path = Path(path)
    tmp = path.with_suffix('.tmp.npz')
    weights = {}
    for i, layer in enumerate((model.layers[0], model.layers[2], model.layers[4]), 1):
        weights[f'w{i}'] = layer.weight.detach().cpu().numpy().T
        weights[f'b{i}'] = layer.bias.detach().cpu().numpy()
    np.savez(tmp, **weights)
    tmp.replace(path)
