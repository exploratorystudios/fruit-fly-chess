import argparse
import chess
from .model import DEFAULT_MODEL, Evaluator
from .search import Search


def main(argv=None):
    p = argparse.ArgumentParser(description='Play against the trained network and CPU search')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--seconds', type=float, default=1)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--fen', help='Analyze this FEN and exit')
    p.add_argument('--color', choices=['white', 'black'], default='white')
    a = p.parse_args(argv)
    engine = Search(Evaluator(a.model))
    if a.fen:
        board = chess.Board(a.fen)
        if not board.is_valid():
            p.error('Invalid chess position')
        result = engine.choose(board, a.seconds, a.depth)
        print(f'move={result.move} score={result.score:.0f}cp depth={result.depth} '
              f'nodes={result.nodes} seconds={result.seconds:.3f}')
        return
    board = chess.Board()
    human = a.color == 'white'
    print('Enter SAN (e4, Nf3) or UCI (e2e4). Commands: quit, undo.')
    while not board.is_game_over(claim_draw=True):
        print('\n' + str(board) + '\n')
        if board.turn == human:
            try:
                text = input('Your move: ').strip()
            except (EOFError, KeyboardInterrupt):
                return
            if text == 'quit':
                return
            if text == 'undo':
                for _ in range(min(2, len(board.move_stack))):
                    board.pop()
                continue
            try:
                board.push_san(text)
            except ValueError:
                print('Illegal or unrecognized move; try again.')
        else:
            result = engine.choose(board, a.seconds, a.depth)
            if result.move is None:
                break
            print(f'Engine: {board.san(result.move)} (depth {result.depth}, {result.nodes} nodes)')
            board.push(result.move)
    print(board)
    print('Result:', board.result(claim_draw=True))


if __name__ == '__main__':
    main()
