"""Glue between the engine and Vercel's Python serverless runtime.

Each file in `api/` is a thin wrapper around `respond()`, so the deployed site
runs exactly the same `dispatch()` as `run.sh site` locally.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastchess.site_server import Engine, dispatch  # noqa: E402

MAX_BODY = 64 * 1024
# One engine per warm container: loading the weights costs more than a search.
_engine = None


def engine():
    global _engine
    if _engine is None:
        weights = ROOT / 'model' / 'fly.npz'
        _engine = Engine(str(ROOT / 'model' / 'best.npz'),
                         seconds=float(os.environ.get('FLY_SECONDS', '1.5')),
                         depth=int(os.environ.get('FLY_DEPTH', '8')),
                         # Keep well inside the function's wall-clock limit.
                         max_seconds=float(os.environ.get('FLY_MAX_SECONDS', '4')))
        # The connectome runs in NumPy, so it deploys alongside the evaluator.
        _engine.fly = None
        if weights.is_file():
            from .fly_numpy import FlyNumpy
            _engine.fly = FlyNumpy(str(weights))
    return _engine
