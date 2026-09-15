"""Local web server for connectome-guided alpha-beta chess.

The default site uses the FlyWire policy to order root moves and the evaluator to
score positions during search. It also serves the networks' real activity to the
visualisation. Binds to localhost only.
"""
import argparse
import json
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import chess
from .model import Evaluator
from .search import Search

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / 'public'
DEFAULT_MODEL = ROOT / 'model' / 'best.npz'
MAX_BODY = 64 * 1024
PIECE_NAMES = ('pawn', 'knight', 'bishop', 'rook', 'queen', 'king')


class Engine:
    def __init__(self, model_path, seconds=1.5, depth=8, max_seconds=8.):
        self.evaluator = Evaluator(model_path)
        self.search = Search(self.evaluator)
        self.fly = None
        self.seconds, self.depth, self.max_seconds = seconds, depth, max_seconds
        # A manifest beside the weights gives the model a real name in the UI.
        manifest = Path(model_path).with_name('manifest.json') if model_path else None
        if manifest and manifest.is_file():
            self.name = json.loads(manifest.read_text()).get('name', manifest.parent.name)
        else:
            self.name = Path(model_path).parent.name if model_path else 'material only'

    def position(self, board):
        """Everything knowable about a position without searching."""
        detail = self.evaluator.inspect(board)
        outcome = board.outcome(claim_draw=True)
        legal = {}
        for move in board.legal_moves:
            legal.setdefault(chess.square_name(move.from_square), []).append({
                'uci': move.uci(), 'to': chess.square_name(move.to_square),
                'promotion': chess.piece_symbol(move.promotion) if move.promotion else None})
        return {
            'fen': board.fen(), 'turn': 'white' if board.turn else 'black',
            'legal': legal, 'check': board.is_check(),
            'moveNumber': board.fullmove_number, 'halfmoveClock': board.halfmove_clock,
            'gameOver': outcome is not None,
            'result': None if outcome is None else outcome.result(),
            'termination': None if outcome is None else outcome.termination.name,
            'lastMove': board.peek().uci() if board.move_stack else None,
            'evaluation': detail,
            'planes': self.planes(board),
        }

    def planes(self, board):
        """The 768 piece features, grouped the way the network sees them:
        12 planes of 64 squares, from the side to move's point of view."""
        flip = 0 if board.turn else 56
        planes = []
        for piece in range(1, 7):
            for opponent in (False, True):
                colour = (not board.turn) if opponent else board.turn
                squares = [(square ^ flip) for square in board.pieces(piece, colour)]
                planes.append({'piece': PIECE_NAMES[piece - 1],
                               'side': 'them' if opponent else 'us',
                               'base': ((piece - 1) * 2 + int(opponent)) * 64,
                               'squares': sorted(squares)})
        return planes

    def candidates(self, board):
        """Static score of every legal move: what the network alone thinks,
        before any look-ahead. Scores are from the mover's point of view."""
        rows = []
        for move in board.legal_moves:
            san = board.san(move)
            board.push(move)
            try:
                rows.append({'uci': move.uci(), 'san': san, 'static': -self.evaluator(board)})
            finally:
                board.pop()
        rows.sort(key=lambda row: row['static'], reverse=True)
        return rows

    def think(self, board, seconds=None, depth=None, guide=True):
        started = time.monotonic()
        budget = seconds or self.seconds
        fly_thought = None
        root_policy = None
        if guide and self.fly is not None and any(board.legal_moves):
            fly_thought = self.fly.think(board, seed=0)
            ranked = fly_thought.get('policy', fly_thought.get('top', []))
            root_policy = {row['uci']: len(ranked) - index
                           for index, row in enumerate(ranked)}
        remaining = max(.01, budget - (time.monotonic() - started))
        result = self.search.choose(board, seconds=remaining,
                                    depth=depth or self.depth, nodes=None,
                                    root_policy=root_policy)
        elapsed = time.monotonic() - started
        pv, probe = [], board.copy()
        # The root move comes from the search result; the rest of the line is
        # recovered by walking the transposition table's stored best moves.
        # (root() deliberately does not write a table entry for the root itself.)
        if result.move is not None:
            pv.append({'uci': result.move.uci(), 'san': probe.san(result.move)})
            probe.push(result.move)
        for _ in range(max(0, (result.depth or 1) - 1)):
            entry = self.search.tt.get(probe._transposition_key())
            move = entry[3] if entry else None
            if move is None or move not in probe.legal_moves:
                break
            pv.append({'uci': move.uci(), 'san': probe.san(move)})
            probe.push(move)
        thought = {
            'move': result.move.uci() if result.move else None,
            'san': board.san(result.move) if result.move else None,
            'score': result.score, 'depth': result.depth, 'nodes': result.nodes,
            'seconds': elapsed, 'nps': result.nodes / elapsed if elapsed else 0,
            'pv': pv, 'ttEntries': len(self.search.tt),
        }
        if fly_thought is not None:
            thought['fly'] = fly_thought
        return thought


def dispatch(engine, action, request):
    """Run one API action. Returns (status, payload); raises nothing for bad input."""
    try:
        board = chess.Board(request.get('fen') or chess.STARTING_FEN)
        if not board.is_valid():
            raise ValueError('invalid position')
    except (ValueError, TypeError) as error:
        return 400, {'error': str(error)}
    try:
        if action == 'position':
            return 200, engine.position(board)
        if action == 'candidates':
            return 200, {'candidates': engine.candidates(board)}
        if action == 'move':
            move = chess.Move.from_uci(request['uci'])
            if move not in board.legal_moves:
                raise ValueError('illegal move')
            board.push(move)
            return 200, engine.position(board)
        if action == 'think':
            # Clamped: a serverless invocation has a hard wall-clock limit.
            seconds = request.get('seconds') or engine.seconds
            seconds = max(.05, min(float(seconds), engine.max_seconds))
            thought = engine.think(board, seconds, request.get('depth'),
                                   guide=request.get('guide') is not False)
            if thought['move']:
                board.push(chess.Move.from_uci(thought['move']))
            return 200, {'thought': thought, 'position': engine.position(board)}
        if action == 'fly-info':
            fly = getattr(engine, 'fly', None)
            return 200, fly.describe() if fly else {'error': 'fly model not loaded'}
        if action == 'fly-observe':
            # Run the connectome on this position without playing its move, so the
            # view can show what it fires even while the evaluator is choosing.
            fly = getattr(engine, 'fly', None)
            if fly is None:
                return 503, {'error': 'fly model not loaded'}
            return 200, {'thought': fly.think(board, request.get('seed'))}
        if action == 'fly-think':
            fly = getattr(engine, 'fly', None)
            if fly is None:
                return 503, {'error': 'start the server with --fly <checkpoint>'}
            thought = fly.think(board, request.get('seed'))
            board.push(chess.Move.from_uci(thought['move']))
            return 200, {'thought': thought, 'position': engine.position(board)}
        if action == 'weights':
            weights = engine.evaluator.weights
            if weights is None:
                return 200, {'w2': [], 'w3': [], 'shape': [0, 0]}
            return 200, {'w2': [round(float(v), 4) for v in weights['w2'].ravel()],
                         'w3': [round(float(v), 4) for v in weights['w3'].ravel()],
                         'shape': list(weights['w2'].shape)}
        if action == 'info':
            return 200, {'model': engine.name, 'seconds': engine.seconds,
                         'depth': engine.depth, 'maxSeconds': engine.max_seconds}
    except (ValueError, KeyError, TypeError) as error:
        return 400, {'error': str(error)}
    return 404, {'error': 'unknown action'}


class Handler(BaseHTTPRequestHandler):
    engine = None

    def log_message(self, *args):
        pass

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        name = 'index.html' if self.path in ('/', '/index.html') else self.path.lstrip('/')
        target = (PUBLIC / name).resolve()
        if not target.is_relative_to(PUBLIC.resolve()) or not target.is_file():
            self.send_error(404)
            return
        body = target.read_bytes()
        kind = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript',
                '.css': 'text/css', '.json': 'application/json'}.get(
                    target.suffix, 'application/octet-stream')
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get('Content-Length') or 0)
        if length > MAX_BODY:
            self.send_json({'error': 'request too large'}, 413)
            return
        try:
            request = json.loads(self.rfile.read(length) or b'{}')
        except json.JSONDecodeError as error:
            self.send_json({'error': str(error)}, 400)
            return
        status, payload = dispatch(self.engine, self.path.rsplit('/', 1)[-1], request)
        self.send_json(payload, status)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=str(DEFAULT_MODEL))
    default_fly = str(ROOT / 'model' / 'fly.npz')
    fly_group = p.add_mutually_exclusive_group()
    fly_group.add_argument('--fly', nargs='?', const=default_fly, default=default_fly,
                           help='Connectome weights (default: model/fly.npz; .npz needs no PyTorch)')
    fly_group.add_argument('--no-fly', dest='fly', action='store_const', const=None,
                           default=argparse.SUPPRESS,
                           help='Disable connectome guidance and use evaluator-only search')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--seconds', type=float, default=1.5)
    p.add_argument('--depth', type=int, default=8)
    a = p.parse_args(argv)
    if not Path(a.model).exists():
        p.error(f'No model at {a.model}')
    Handler.engine = Engine(a.model, a.seconds, a.depth)
    Handler.engine.fly = None
    if a.fly:
        if not Path(a.fly).is_file():
            p.error(f'No fly model at {a.fly}')
        print('loading the fly connectome…', flush=True)
        if a.fly.endswith('.npz'):
            from .fly_numpy import FlyNumpy
            Handler.engine.fly = FlyNumpy(a.fly)
        else:
            from .fly_engine import FlyEngine     # the PyTorch path, for comparison
            Handler.engine.fly = FlyEngine(a.fly)
        info = Handler.engine.fly.describe()
        print(f"fly: {info['neurons']:,} neurons, {info['synapses']:,} synapses, "
              f"{info['timesteps']} timesteps, {info.get('runtime', 'torch')} runtime",
              flush=True)
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f'Fly Chess — model {Handler.engine.name}, {a.seconds}s/move, max depth {a.depth}')
    print(f'Open http://{a.host}:{a.port}/   (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')


if __name__ == '__main__':
    main()
