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
        _engine = Engine(str(ROOT / 'model' / 'best.npz'),
                         seconds=float(os.environ.get('FLY_SECONDS', '1.5')),
                         depth=int(os.environ.get('FLY_DEPTH', '8')),
                         # Keep well inside the function's wall-clock limit.
                         max_seconds=float(os.environ.get('FLY_MAX_SECONDS', '8')))
    return _engine


def respond(request_handler, action):
    """Read a JSON body, run one action, write the JSON response."""
    try:
        length = int(request_handler.headers.get('Content-Length') or 0)
    except (TypeError, ValueError):
        length = 0
    if length > MAX_BODY:
        status, payload = 413, {'error': 'request too large'}
    else:
        try:
            body = json.loads(request_handler.rfile.read(length) or b'{}')
            status, payload = dispatch(engine(), action, body)
        except json.JSONDecodeError as error:
            status, payload = 400, {'error': str(error)}
    encoded = json.dumps(payload).encode()
    request_handler.send_response(status)
    request_handler.send_header('Content-Type', 'application/json')
    request_handler.send_header('Content-Length', str(len(encoded)))
    request_handler.send_header('Cache-Control', 'no-store')
    request_handler.end_headers()
    request_handler.wfile.write(encoded)
