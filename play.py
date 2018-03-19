"""Play chess using random_players."""

import chess
import random
from chai.ai import ChessAI

def calc_reward(board):
    """Returns the numerical reward for the game result."""
    rslt = str(board.result())
    print(rslt)
    if rslt == '1-0':
        return +1.0
    elif rslt == '0-1':
        return -1.0
    elif rslt == '1/2-1/2':
        return 0.0
    else:
        raise AssertionError('Wrong board result: %s' % rslt)

player = ChessAI(piece_emb_size=20, lr=.001)

num_games = 100  # number of games to train the AI for
verbose = True

# for each game
for train_index in range(num_games):

    board = chess.Board()

    map = board.piece_map()

    print(map)
    print(board.mirror())

    train_examples = []
    while not board.is_game_over():

        moves = [move for move in board.legal_moves]  # legal moves
        boards = []  # Resultant boards from all legal moves

        # Get resultant board positions
        for move in moves:
            rslt_board = board.copy()
            rslt_board.push(move)
            boards.append(rslt_board)

        # tell player to make a move (choose a board)
        choice_index = player.make_move(boards)

        board = boards[choice_index]

        train_examples.append((boards, choice_index))

    reward = calc_reward(board)
    player.reinforce(train_examples, reward)

    if verbose:
        print(board)
        print('Checkmate: %s' % board.is_checkmate())
        print(board.result())
        print("White's turn: %s" % board.turn)
        print('Stalemate: %s' % board.is_stalemate())
        print('Five-fold repetition: %s' % board.is_fivefold_repetition())
        print('Insufficient material: %s' % board.is_insufficient_material())
