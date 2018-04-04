"""Play chess using random_players."""

import chess
import os
from chai.ai import ChessAI, RandomPlayer
import tqdm
import random
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def calc_result(board):
    """Returns the numerical reward for the game result."""
    rslt = str(board.result())
    if rslt == '1-0':
        return 1
    elif rslt == '0-1':
        return 0
    elif rslt == '1/2-1/2':
        return 0.5
    else:
        raise AssertionError('Wrong board result: %s' % rslt)

save_dir = os.path.join(DATA_DIR, 'models/')
num_games = 100  # number of games to train the AI for
games_per_save = 10000
games_per_print = 10000
verbose = True
restore= True
test = True  # if test, then black will be a random player.

print('Chess AI Version 0.1')

# make save directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

white = ChessAI(piece_emb_size=100, lr=0.01, save_dir=save_dir, restore=restore)

if test:
    black = RandomPlayer()  # testing
else:
    black = white  # training

white_wins = 0
black_wins = 0
draws = 0

loss_values = []  # value of loss function over multiple games
scores = []
abs_scores = []
avg_game_lens = []
accuracies = []

# for each game
for train_index in tqdm.tqdm(range(num_games)):

    board = chess.Board()

    # map = board.piece_map()

    while not board.is_game_over():

        moves = [move for move in board.legal_moves]

        if test:
            # Get board positions resulting from all legal moves

            # tell player to make a move (choose a board)
            if board.turn:
                choice_index = white.make_move(board, test=test)
            else:
                choice_index = black.make_move(board, test=test)

            # think about changing this statement
            board.push(moves[choice_index])
        else:
            rand_index = random.sample(range(len(moves)), 1)[0]

            board.push(moves[rand_index])

        if not board.is_valid():
            raise ValueError('Boards is invalid: %s\n' % str(board))

    result = calc_result(board)

    # Record wins and draws
    if result == 1:
        white_wins += 1
    elif result == 0:
        black_wins += 1
    elif result == 0.5:
        draws += 1
    else:
        print('Wrong result!')

    # Gather statistics
    avg_game_lens.append(len(board.move_stack))

    # Train on game
    loss = None

    if not test and result != 0.5:
        # we give the game board to train on!
        loss, game_scores, accuracy = white.reinforce(board, result)
        accuracies.append(accuracy)
        if game_scores is not None:
            scores.append(np.mean(game_scores))
            abs_scores.append(np.mean(np.abs(game_scores)))
        loss_values.append(loss)

    # Save periodically and print out stats
    if train_index % games_per_print == 0 and train_index != 0:
        print('Saving. Printing stats...')
        print('White wins: %s' % white_wins)
        print('Black wins: %s' % black_wins)
        print('Draws: %s' % draws)

        print('Avg game len: %s' % np.mean(avg_game_lens))
        avg_game_lens = []

        print('Avg loss: %s' % np.mean(loss_values))
        if len(scores) > 0:
            print('Avg score: %s' % np.mean(scores))
            print('Avg abs score: %s' % np.mean(abs_scores))
            print('Accuracy: %s' % np.mean(accuracies))
            accuracies = []
            loss_values = []
            scores = []

    if train_index % games_per_save == 0 and train_index != 0 and not test:
        white.save()

    # Print per game statistics
    if verbose:
        print(board)
        print('Reward: %s' % result)
        print('Checkmate: %s' % board.is_checkmate())
        print(board.result())
        print("White's turn: %s" % board.turn)
        print('Stalemate: %s' % board.is_stalemate())
        print('Five-fold repetition: %s' % board.is_fivefold_repetition())
        print('Insufficient material: %s' % board.is_insufficient_material())
        print('Seventy-five Moves: %s' % board.is_seventyfive_moves())
        if loss is not None:
            print('Loss: %s' % loss)


# Print total win statistics
print('Total statistics:')
print('Number of games: %s' % num_games)
print('Number of white wins: %s' % white_wins)
print('Number of black wins: %s' % black_wins)
print('Number of draws: %s' % draws)
