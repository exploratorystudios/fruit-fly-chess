"""Measure games against material-only search, the frozen baseline search,
random play, or Stockfish. Reports a score, not an Elo estimate."""
import argparse
import hashlib
import json
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import chess
import chess.engine
from .model import DEFAULT_MODEL, Evaluator
from .search import Search
from .search_baseline import Search as BaselineSearch


class FromScratch:
    """Wraps the evaluator so it is not an Evaluator instance, which makes Search
    fall back to rebuilding the first layer at every leaf. Used to A/B the
    incremental accumulator with everything else held fixed."""

    def __init__(self, path):
        self.inner = Evaluator(path)

    def __call__(self, board):
        return self.inner(board)

OPENINGS = [[], ['e2e4', 'e7e5', 'g1f3', 'b8c6'],
            ['d2d4', 'd7d5', 'c2c4', 'e7e6'], ['e2e4', 'c7c5', 'g1f3', 'd7d6'],
            ['d2d4', 'g8f6', 'c2c4', 'g7g6'], ['c2c4', 'e7e5', 'b1c3', 'g8f6']]


def play_game(index, settings):
    """Play one game and return its record. Runs in a worker process."""
    a = argparse.Namespace(**settings)
    ours = Search(Evaluator(a.model))
    # 'material' swaps the network for material-only eval; 'old-search' keeps the
    # same network and swaps the search, isolating one change at a time.
    if a.opponent == 'material':
        rival = Search(Evaluator())
    elif a.opponent == 'old-search':
        rival = BaselineSearch(Evaluator(a.model))
    elif a.opponent == 'no-accumulator':
        rival = Search(FromScratch(a.model))
    elif a.opponent == 'model':
        rival = Search(Evaluator(a.rival_model))
    elif a.opponent == 'no-pruning':
        rival = Search(Evaluator(a.model), pruning=False)
    else:
        rival = None
    teacher = chess.engine.SimpleEngine.popen_uci(a.stockfish) if a.opponent == 'stockfish' else None
    if teacher:
        teacher.configure({'Threads': 1, 'Hash': 32})
    rng = random.Random(42 + index)
    board = chess.Board()
    openings = getattr(a, 'openings', OPENINGS)
    for move in openings[(index // 2) % len(openings)]:
        board.push_uci(move)
    color = index % 2 == 0
    played, depths, rival_depths, node_counts, rival_nodes = 0, [], [], [], []
    try:
        while not board.is_game_over(claim_draw=True) and played < a.max_plies:
            if board.turn == color:
                result = ours.choose(board, a.seconds, a.depth, getattr(a, 'nodes', 1000000))
                move = result.move
                depths.append(result.depth)
                node_counts.append(result.nodes)
            elif teacher:
                move = teacher.play(board, chess.engine.Limit(nodes=a.stockfish_nodes), game=index).move
            elif a.opponent == 'random':
                move = rng.choice(list(board.legal_moves))
            else:
                result = rival.choose(board, a.seconds, a.depth, getattr(a, 'nodes', 1000000))
                move = result.move
                rival_depths.append(result.depth)
                rival_nodes.append(result.nodes)
            if move is None:
                break
            board.push(move)
            played += 1
    finally:
        if teacher:
            teacher.quit()
    outcome = board.outcome(claim_draw=True)
    status = ('unfinished' if outcome is None else 'draw' if outcome.winner is None
              else 'win' if outcome.winner == color else 'loss')
    return dict(game=index + 1, color='white' if color else 'black', status=status,
                plies=played, final_fen=board.fen(),
                mean_depth=sum(depths) / len(depths) if depths else 0,
                rival_mean_depth=sum(rival_depths) / len(rival_depths) if rival_depths else 0,
                mean_nodes=sum(node_counts) / len(node_counts) if node_counts else 0,
                rival_mean_nodes=sum(rival_nodes) / len(rival_nodes) if rival_nodes else 0,
                moves=[m.uci() for m in board.move_stack])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--opponent',
                   choices=['material', 'model', 'old-search', 'no-accumulator',
                            'no-pruning', 'random', 'stockfish'],
                   default='material')
    p.add_argument('--rival-model', help='Opposing weights for --opponent model')
    p.add_argument('--games', type=int, default=10)
    p.add_argument('--seconds', type=float, default=.1, help='Equal budget per move for both sides')
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--nodes', type=int, default=1000000, help='Equal node cap per move')
    p.add_argument('--fixed-nodes', action='store_true', help='Disable the clock; use --nodes and --depth')
    p.add_argument('--openings-file', help='JSON list of UCI move lists, paired with colors swapped')
    p.add_argument('--max-plies', type=int, default=240)
    p.add_argument('--parallel', type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument('--stockfish-nodes', type=int, default=1000)
    p.add_argument('--stockfish', default=shutil.which('stockfish') or '/usr/games/stockfish')
    p.add_argument('--out', default='runs/evaluation.json')
    a = p.parse_args(argv)
    if min(a.games, a.seconds, a.depth, a.nodes, a.max_plies, a.stockfish_nodes, a.parallel) <= 0:
        p.error('budgets must be positive')
    if a.opponent == 'model' and not a.rival_model:
        p.error('--opponent model needs --rival-model')
    if a.fixed_nodes:
        a.seconds = None
    settings, records = vars(a).copy(), {}
    if a.openings_file:
        try:
            openings = json.loads(Path(a.openings_file).read_text())
            if not isinstance(openings, list) or not openings:
                raise ValueError('expected a nonempty list of move lists')
            for line in openings:
                if not isinstance(line, list):
                    raise ValueError('each opening must be a move list')
                board = chess.Board()
                for move in line:
                    if not isinstance(move, str):
                        raise ValueError('opening moves must be UCI strings')
                    board.push_uci(move)
                if board.is_game_over(claim_draw=True):
                    raise ValueError('opening ends in a terminal position')
            settings['openings'] = openings
        except (OSError, ValueError, TypeError) as error:
            p.error(f'Invalid openings file: {error}')
    started = time.monotonic()
    if a.parallel == 1:
        for i in range(a.games):
            records[i] = play_game(i, settings)
            print(f'game {i+1}/{a.games}: {records[i]["status"]}, {records[i]["plies"]} plies', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.parallel) as pool:
            futures = {pool.submit(play_game, i, settings): i for i in range(a.games)}
            for future in as_completed(futures):
                i = futures[future]
                records[i] = future.result()
                print(f'game {i+1}/{a.games}: {records[i]["status"]}, {records[i]["plies"]} plies, '
                      f'mean depth {records[i]["mean_depth"]:.1f} | done {len(records)}/{a.games}', flush=True)
    ordered = [records[i] for i in sorted(records)]
    counts = {s: sum(r['status'] == s for r in ordered) for s in ('win', 'draw', 'loss', 'unfinished')}
    report = dict(settings=settings, counts=counts, parallel=a.parallel,
                  seconds=time.monotonic()-started, games=ordered)
    report['model_sha256'] = hashlib.sha256(Path(a.model).read_bytes()).hexdigest()
    if a.opponent == 'model':
        report['rival_model_sha256'] = hashlib.sha256(Path(a.rival_model).read_bytes()).hexdigest()
    report['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ('model.py', 'accumulator.py', 'search.py', 'evaluate.py')}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(counts))
    print(f'Saved {out} in {report["seconds"]:.0f}s. Unfinished games are not counted as draws; '
          'these results are not Elo.')


if __name__ == '__main__':
    main()
