"""Play games and calculate statistics."""

import chess
import random
import tqdm

verbose = False

num_games = 10000

num_white = 0
num_black = 0
num_draw = 0

for game_index in tqdm.tqdm(range(num_games)):

    board = chess.Board()

    while not board.is_game_over():
        # Make random moves until game over
        moves = [move for move in board.legal_moves]
        rand_index = random.sample(range(len(moves)), 1)[0]
        board.push(moves[rand_index])

    if verbose:
        print(board)
        print('Checkmate: %s' % board.is_checkmate())
        print(board.result())
        print("White's turn: %s" % board.turn)
        print('Stalemate: %s' % board.is_stalemate())
        print('Five-fold repetition: %s' % board.is_fivefold_repetition())
        print('Insufficient material: %s' % board.is_insufficient_material())

    rslt = str(board.result())

    if rslt == '1/2-1/2':
        num_draw += 1
    if rslt == '1-0':
        num_white += 1
    if rslt == '0-1':
        num_black += 1

print('Num white wins: %s' % num_white)
print('Num black wins: %s' % num_black)
print('Num draws: %s' % num_draw)
