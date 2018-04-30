"""Allow human to play games against AI"""
import chess
import os
from chai.ai import ChessAI, HumanPlayer

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
save_dir = os.path.join(DATA_DIR, 'models/')

emb_size = 100
board = chess.Board()

ai = ChessAI(piece_emb_size=emb_size, lr=0.01, save_dir=save_dir, restore=True)

human = HumanPlayer()

while not board.is_game_over():

    moves = [move for move in board.legal_moves]

    if board.turn:
        choice_index = human.make_move(board)
    else:
        choice_index = ai.make_move(board, test=True)

    board.push(moves[choice_index])

print(board)
print('Checkmate: %s' % board.is_checkmate())
print(board.result())
print("White's turn: %s" % board.turn)
print('Stalemate: %s' % board.is_stalemate())
print('Five-fold repetition: %s' % board.is_fivefold_repetition())
print('Insufficient material: %s' % board.is_insufficient_material())
print('Seventy-five Moves: %s' % board.is_seventyfive_moves())


