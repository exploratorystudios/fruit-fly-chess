"""Clocked, paired-opening match against Stockfish at a chosen UCI_Elo setting.

Games are independent, so they run in parallel worker processes by default;
that is what makes a ten-game match finish in minutes rather than hours.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import chess
import chess.engine
import chess.pgn
from .evaluate import OPENINGS
from .model import DEFAULT_MODEL, Evaluator
from .search import Search


def rating_summary(counts, anchor):
    n = counts['win'] + counts['draw'] + counts['loss']
    if not n:
        return None
    score = counts['win'] + .5 * counts['draw']
    p = score / n

    def elo(prob):
        return anchor + 400 * math.log10(prob / (1 - prob)) if 0 < prob < 1 else None
    z = 1.959963984540054
    # Per-game scores are 1 / 0.5 / 0, so the spread comes from the actual
    # win-draw-loss mix. A binomial interval ignores that draws carry half a
    # point and no variance, and is badly too wide for draw-heavy matches.
    variance = (counts['win'] + .25 * counts['draw']) / n - p * p
    if variance > 0:
        radius = z * math.sqrt(variance / n)
        low, high = p - radius, p + radius
        method = 'normal approximation on the mean game score (win/draw/loss variance)'
    else:
        # Every game scored identically: fall back to the Wilson interval.
        center = (p + z*z/(2*n)) / (1+z*z/n)
        radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
        low, high = center - radius, center + radius
        method = 'Wilson score interval (no variation in game scores)'
    # A bound that runs past a 0% or 100% score has no finite Elo. Say so, rather
    # than emitting a bare null that reads like a failed computation.
    bounds = [elo(max(0., low)), elo(min(1., high))]
    unbounded = [side for side, value in zip(('lower', 'upper'), bounds) if value is None]
    result = dict(games=n, points=score, score_fraction=p, opponent_nominal_elo=anchor,
                  performance_elo=elo(p), interval_method=method,
                  approximate_95pct_interval=bounds,
                  caveat='Nominal Stockfish-relative performance, not FIDE/Chess.com/Lichess Elo. '
                         'Small sample, paired openings, stochastic opponent; interval assumes '
                         'independent games and a fixed accurate anchor.')
    if unbounded:
        result['interval_note'] = (
            f'The {" and ".join(unbounded)} bound reaches a {"0%" if "lower" in unbounded else "100%"}'
            ' score, where no finite Elo exists, so it is reported as null (unbounded), not zero.')
    if score == 0:
        result['one_sided_95pct_upper_elo'] = elo(1 - .05 ** (1/n))
        result['performance_elo'] = None
        result['explanation'] = ('Zero points gives no finite maximum-likelihood Elo estimate; '
                                 'report an upper bound rather than inventing a point rating.')
    elif score == n:
        result['one_sided_95pct_lower_elo'] = elo(.05 ** (1/n))
        result['performance_elo'] = None
    return result


def stockfish_options(engine, elo):
    """Weakest supported play unless a higher UCI_Elo is requested."""
    option = engine.options['UCI_Elo']
    target = option.min if elo is None else max(option.min, min(option.max, elo))
    options = {'Threads': 1, 'Hash': 32, 'UCI_LimitStrength': True, 'UCI_Elo': target}
    if target == option.min:
        options['Skill Level'] = 0
    return options


def play_game(index, settings):
    """Play one game and return its record plus PGN. Runs in a worker process."""
    a = argparse.Namespace(**settings)
    ours = Search(Evaluator(a.model))
    board = chess.Board()
    opening = OPENINGS[(index // 2) % len(OPENINGS)]
    for move in opening:
        board.push_uci(move)
    color = index % 2 == 0
    clocks = {chess.WHITE: a.clock, chess.BLACK: a.clock}
    played, logs, winner, termination = 0, [], None, None
    began_game = time.monotonic()
    with chess.engine.SimpleEngine.popen_uci(a.stockfish) as teacher:
        options = stockfish_options(teacher, a.opponent_elo)
        teacher.configure(options)
        name = teacher.id['name']
        while played < a.max_plies:
            outcome = board.outcome(claim_draw=True)
            if outcome:
                winner, termination = outcome.winner, outcome.termination.name
                break
            turn = board.turn
            began = time.monotonic()
            detail = {}
            if turn == color:
                budget = None if a.untimed_fast else max(.001, min(clocks[turn] * .9,
                                                                   clocks[turn]/30 + .8*a.increment))
                result = ours.choose(board, seconds=budget, depth=a.depth,
                                     nodes=None if a.untimed_fast else a.max_nodes)
                move = result.move
                detail = dict(depth=result.depth, nodes=result.nodes,
                              score_cp=result.score, budget=budget)
            else:
                result = teacher.play(board, chess.engine.Limit(
                    white_clock=clocks[chess.WHITE], black_clock=clocks[chess.BLACK],
                    white_inc=a.increment, black_inc=a.increment), game=index)
                move = result.move
            elapsed = time.monotonic() - began
            untimed = a.untimed_fast and turn == color
            if not untimed:
                clocks[turn] -= elapsed
                if clocks[turn] <= 0:
                    winner = None if board.has_insufficient_material(not turn) else not turn
                    termination = 'TIME_FORFEIT'
                    break
            if move is None or move not in board.legal_moves:
                raise RuntimeError(f'Engine returned illegal/no move in nonterminal position: {move}')
            if not untimed:
                clocks[turn] += a.increment
            logs.append(dict(move=move.uci(), seconds=elapsed,
                             clock=None if untimed else clocks[turn], **detail))
            board.push(move)
            played += 1
        # Check the final move even if it landed exactly on the cap.
        if termination is None:
            outcome = board.outcome(claim_draw=True)
            if outcome:
                winner, termination = outcome.winner, outcome.termination.name
        status = ('unfinished' if termination is None else 'draw' if winner is None
                  else 'win' if winner == color else 'loss')
        result_text = ('*' if termination is None else '1/2-1/2' if winner is None
                       else '1-0' if winner else '0-1')
        game = chess.pgn.Game.from_board(board)
        game.headers.update(Event=f'Fast Chess vs {name} (UCI_Elo {options["UCI_Elo"]})',
                            Round=str(index + 1),
                            White='Fast Chess' if color else name,
                            Black=name if color else 'Fast Chess',
                            TimeControl=f'{a.clock:g}+{a.increment:g}',
                            Result=result_text, Termination=termination or 'move cap')
        if a.untimed_fast:
            game.headers['TimeControl'] = '?'
            game.headers['FastChessTimeControl'] = f'unlimited time, depth {a.depth}'
            game.headers['StockfishTimeControl'] = f'{a.clock:g}+{a.increment:g}'
        node = game
        for ply, move in enumerate(board.move_stack):
            node = node.variations[0]
            if ply >= len(opening):
                entry = logs[ply - len(opening)]
                if entry['clock'] is not None:
                    node.set_clock(entry['clock'])
        ours_moves = [entry for entry in logs if 'depth' in entry]
        record = dict(game=index + 1, color='white' if color else 'black', status=status,
                      result=result_text, termination=termination or 'MOVE_CAP',
                      opening=opening, plies=played, final_fen=board.fen(),
                      moves=[m.uci() for m in board.move_stack], timing=logs,
                      seconds=time.monotonic() - began_game,
                      stockfish_options=options, stockfish_id=dict(teacher.id),
                      mean_depth=(sum(m['depth'] for m in ours_moves) / len(ours_moves)
                                  if ours_moves else 0),
                      mean_seconds_per_move=(sum(m['seconds'] for m in ours_moves) / len(ours_moves)
                                             if ours_moves else 0))
    return record, str(game)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--games', type=int, default=10)
    p.add_argument('--clock', type=float, default=60, help='Seconds per side; both engines share it')
    p.add_argument('--increment', type=float, default=.6)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--max-nodes', type=int, default=100000000)
    p.add_argument('--opponent-elo', type=int, default=None,
                   help="Stockfish UCI_Elo target; default is the binary's minimum")
    p.add_argument('--untimed-fast', action='store_true',
                   help='Let Fast Chess finish the configured depth without clock or node limits')
    p.add_argument('--parallel', type=int, default=min(4, os.cpu_count() or 1),
                   help='Games to play concurrently; 1 reproduces a single-game-at-a-time match')
    p.add_argument('--max-plies', type=int, default=600)
    p.add_argument('--stockfish', default=shutil.which('stockfish') or '/usr/games/stockfish')
    p.add_argument('--out', default='runs/stockfish-weakest-10')
    a = p.parse_args(argv)
    if min(a.games, a.clock, a.depth, a.max_plies, a.parallel, a.max_nodes) <= 0 or a.increment < 0:
        p.error('Positive budgets required; increment must be nonnegative')
    out = Path(a.out)
    if out.with_suffix('.json').exists():
        p.error('Output already exists; select another --out to preserve the previous match')
    out.parent.mkdir(parents=True, exist_ok=True)

    settings = vars(a).copy()
    report = dict(settings=settings, parallel=a.parallel,
                  model_sha256=hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
                  source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('model.py', 'search.py', 'features.py', 'benchmark.py')},
                  games=[])
    records, pgns = {}, {}
    started = time.monotonic()

    def save():
        ordered = [records[i] for i in sorted(records)]
        counts = {s: sum(r['status'] == s for r in ordered) for s in ('win', 'draw', 'loss', 'unfinished')}
        anchor = ordered[0]['stockfish_options']['UCI_Elo'] if ordered else None
        report.update(games=ordered, counts=counts, completed_games=len(ordered),
                      seconds=time.monotonic() - started,
                      rating=rating_summary(counts, anchor) if anchor else None)
        if report['rating'] and a.untimed_fast:
            report['rating']['time_handicap'] = (
                'Fast Chess has unlimited wall time to complete the configured depth; Stockfish '
                'uses the stated clock. This is an unequal-time performance comparison.')
        if report['rating'] and a.parallel > 1:
            report['rating']['parallel_note'] = (
                f'{a.parallel} games shared this machine, so each engine had less CPU than in a '
                'solo match. Both sides are affected, but a clocked result is not identical to a '
                'one-game-at-a-time run.')
        tmp = out.with_suffix('.tmp.json')
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(out.with_suffix('.json'))
        out.with_suffix('.pgn').write_text('\n\n'.join(pgns[i] for i in sorted(pgns)) + '\n')

    print(f'{a.games} games, {a.parallel} at a time, {a.clock:g}+{a.increment:g}, '
          f'Fast Chess depth {a.depth}{" (untimed)" if a.untimed_fast else ""}', flush=True)
    if a.parallel == 1:
        for i in range(a.games):
            records[i], pgns[i] = play_game(i, settings)
            save()
            print(f'Game {i+1}/{a.games}: {records[i]["status"]} ({records[i]["termination"]}), '
                  f'cumulative {report["counts"]}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.parallel) as pool:
            futures = {pool.submit(play_game, i, settings): i for i in range(a.games)}
            for future in as_completed(futures):
                i = futures[future]
                records[i], pgns[i] = future.result()
                save()
                print(f'Game {i+1}/{a.games}: {records[i]["status"]} ({records[i]["termination"]}), '
                      f'{records[i]["plies"]} plies, mean depth {records[i]["mean_depth"]:.1f}, '
                      f'{records[i]["seconds"]:.0f}s | done {len(records)}/{a.games} '
                      f'{report["counts"]}', flush=True)
    print(json.dumps(report['rating'], indent=2))
    print(f'Saved {out}.json and {out}.pgn in {report["seconds"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
