"""Scale an exported network's correction to material without changing its cost.

Choose the scale using development matches, then verify it in separate matches.
Static validation loss alone does not select a good search evaluator.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .model import Evaluator


def calibrate(source, destination, scale):
    if not np.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError('scale must be finite and between 0 and 1')
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError(f'{destination} already exists; choose another output')
    weights = Evaluator(source).weights
    # The residual is 100 * (hidden @ w3 + b3); material lives outside the net.
    weights['w3'] *= scale
    weights['b3'] *= scale
    metadata = dict(source=str(source.resolve()),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    correction_scale=scale)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix('.tmp.npz')
    np.savez(tmp, **weights, calibration=json.dumps(metadata))
    tmp.replace(destination)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--scale', required=True, type=float)
    a = p.parse_args(argv)
    try:
        calibrate(a.model, a.out, a.scale)
    except ValueError as error:
        p.error(str(error))
    print(f'Saved {a.out}; material + {a.scale:g} * learned correction. Validate in games before using.')


if __name__ == '__main__':
    main()
