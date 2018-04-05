"""Chess AI made in Tensorflow."""
import tensorflow as T
import random
import numpy as np
import chess

piece_ids = {'r': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5, 'p': 6,
             'R': 7, 'N': 8, 'B': 9, 'Q': 10, 'K': 11, 'P': 12, ' ': 0}

class ChessAI:
    def __init__(self, piece_emb_size, lr, restore=False, save_dir=None, **kwargs):
        self.piece_emb_size = piece_emb_size

        self.save_dir = save_dir
        self.restore = restore
        self.lr = lr  # learning rate
        self.build()

        # Create session and model
        self.config = T.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init = T.global_variables_initializer()
        self.sess = T.InteractiveSession(config=self.config)
        self.sess.run(self.init)
        self.saver = T.train.Saver(var_list=T.trainable_variables(), max_to_keep=10)
        self.step = 0

        if restore and save_dir is not None:
            restore_model_from_save(save_dir, self.sess)

    def build(self):
        """Build convolutional neural network."""

        num_pieces = 13  # 6 piece types * 2 sides + 1 for empty square

        # Input board configuration, each element an index corresponding to a chess piece type at that position
        boards = T.placeholder(T.int32, shape=(None, 8, 8), name='boards')
        # tries to predict label results - whether or not white will win the game
        results = T.placeholder(T.float32, shape=(None,), name='result')

        turn = T.placeholder(T.float32, shape=(None,), name='turn')

        # add turn as feature to each square of board
        turn_per_square = T.tile(T.reshape(turn, (-1, 1, 1, 1)), [1, 8, 8, 1])

        piece_embs = T.get_variable('embs', shape=(num_pieces, self.piece_emb_size),
                                    initializer=T.contrib.layers.xavier_initializer(uniform=False))

        # neural representation of board
        board_embs = T.nn.embedding_lookup(piece_embs, boards)

        # we give the board and whoever's turn it is as input
        input_embs = T.concat([board_embs, turn_per_square], axis=-1)

        layer = input_embs

        conv1 = T.layers.conv2d(
            inputs=board_embs,
            filters=self.piece_emb_size+1,
            kernel_size=[2, 2],
            padding="same",
            activation=T.nn.tanh)

        layer = layer + conv1

        # conv2 = T.layers.conv2d(
        #     inputs=layer,
        #     filters=self.piece_emb_size+1,
        #     kernel_size=[4, 4],
        #     padding="same",
        #     activation=T.nn.tanh)
        #
        # layer = layer + conv2

        # flatten features and scale values
        layer_flat = T.contrib.layers.flatten(layer)
        layer_flat = layer_flat / int(layer_flat.get_shape()[1])

        scores = build_linear_layer('linear', layer_flat, 1, xavier=True)
        scores = T.reshape(scores, shape=(-1,), name='scores')
        probs = T.nn.sigmoid(scores)

        self.loss = T.reduce_mean(T.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=results))

        self.train = T.train.AdamOptimizer(self.lr).minimize(self.loss)

        # input each board position, and (for training) whether playing from that position resulted in a win
        self.i = {'boards': boards, 'results': results, 'turn': turn}
        # outputs the probability of winning from each board position, the highest probability board position,
        # and the loss function with respect to the label
        self.o = {'probs': probs, 'scores': scores, 'loss': self.loss}

    def make_move(self, board, test=False, k=0.5):
        """Given boards corresponding to each move,
        returns the index of the board move the AI choses.

        boards - resulting boards after making every legal move
        k - temperature constant of softmax"""

        np_board_list = []

        for move in board.legal_moves:
            board.push(move)
            np_board = board_to_numpy(board)
            np_board_list.append(np_board)
            board.pop()

        whites_turn = board.turn

        np_turn = np.repeat(not whites_turn, len(np_board_list), axis=0)

        np_boards = np.stack(np_board_list, axis=0)

        scores = self.sess.run(self.o['scores'], feed_dict={self.i['boards']: np_boards,
                                                            self.i['turn']: np_turn})

        # scores tell us how likely white is to win the game from each board
        #

        # invert scores if it is black's turn
        if not whites_turn:
            scores = -scores

        if test:
            return np.argmax(scores)
        else:
            # select from top n_select scoring board positions
            n_select = int(k * np_boards.shape[0])
            max_indices = np.argpartition(scores, -n_select)[-n_select:]
            choice_index = random.sample(list(max_indices), 1)[0]
            return choice_index


    def reinforce(self, board, result):
        """Apply reinforcement learning signal to AI.

        boards - board positions seen throughout a single game of chess
        result - whether white won or lost

        Returns: loss function value"""

        # okay, we start with the board
        # the board has a stack
        # replay each move and convert to np_board

        num_moves = len(board.move_stack)

        np_results = np.repeat(result, num_moves + 1, axis=0) # + 1 for starting board position

        # input who's turn it is (0 - black, 1 - white)
        np_turn = np.zeros([num_moves + 1])

        replay_board = chess.Board()

        np_board_list = []

        for move_index in range(num_moves + 1):

            np_board = board_to_numpy(replay_board)
            np_board_list.append(np_board)

            np_turn[move_index] = replay_board.turn

            # on last iteration, there is no move to make
            if move_index != num_moves:
                replay_board.push(board.move_stack[move_index])

        np_boards = np.stack(np_board_list, axis=0)

        # we predict probability of white winning the game

        loss, scores, _ = self.sess.run([self.loss, self.o['scores'], self.train],
                                feed_dict={self.i['results']: np_results,
                                           self.i['boards']: np_boards,
                                           self.i['turn']: np_turn})

        probs = 1 / (1 + np.exp(-scores))
        preds = np.round(probs)
        wrong_preds = np.bitwise_xor(preds.astype(bool), np_results.astype(bool))
        accuracy = 1 - np.mean(wrong_preds)

        return loss, scores, accuracy

    def save(self):
        assert self.save_dir is not None
        self.saver.save(self.sess, self.save_dir, global_step=self.step, write_meta_graph=False)
        self.step += 1


def board_to_numpy(board, mirror=False):
    """Converts chess boards to numpy format. Each
    chess board becomes 8x8 matrix of ids, each id
    corresponding to each piece in chess (6 pieces for
    white and 6 pieces for black).

    mirror - white's pieces become black's pieces and visa versa
             board positions are mirrored vertically

    Returns: m x 8 x 8 numpy array for m boards.
    0 represents empty space, 1-6 represent
    (rook, knight, bishop, queen, king, pawn)
    for white and 7-12 for black."""

    np_board = np.zeros((8, 8))
    map = board.piece_map()

    # Fill in board with pieces
    for position in map:
        piece = str(map[position])
        y = position // 8
        x = position % 8

        if mirror:
            piece.swapcase()
            y = 7 - y

        np_board[x, y] = piece_ids[piece]

    return np_board


def build_linear_layer(name, input_tensor, output_size, xavier=False):
    """Build linear layer by creating random weight matrix and bias vector,
    and applying them to input. Weights initialized with random normal
    initializer.

    Arguments:
        - name: Required for unique Variable names
        - input: (num_examples x layer_size) matrix input
        - output_size: size of output Tensor

    Returns: Output Tensor of linear layer with size (num_examples, out_size).
        """

    if xavier:
        initializer = T.contrib.layers.xavier_initializer(uniform=False)
    else:
        initializer = T.random_normal_initializer(stddev=0.00001)

    input_size = input_tensor.get_shape()[-1]  #tf.shape(input_tensor)[1]
    with T.variable_scope(name):
        scale_w = T.get_variable('w', shape=(input_size, output_size),
                                 initializer= initializer) # tf.contrib.layers.xavier_initializer(uniform=False)) #

        scale_b = T.get_variable('b', shape=(output_size,), initializer=T.zeros_initializer())

    return T.matmul(input_tensor, scale_w) + scale_b


def restore_model_from_save(model_var_dir, sess, var_list=None):
    """Restores all model variables from the specified directory."""
    if var_list is None:
        var_list = T.trainable_variables()
    saver = T.train.Saver(max_to_keep=10, var_list=var_list)
    # Restore model from previous save.
    ckpt = T.train.get_checkpoint_state(model_var_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restoring from save')
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("No checkpoint found!")
        return -1


def load_scope_from_save(save_dir, sess, scope):
    """Load the encoder model variables from checkpoint in save_dir.
    Store them in session sess."""
    variables = T.get_collection(T.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    if not len(variables) > 0:
        raise AssertionError('Load scope %s must contain trainable variables!' % str(scope))
    restore_model_from_save(save_dir, sess, var_list=variables)


class RandomPlayer:
    """Player which makes random moves. This is the baseline against which
    we will compare our chess AI."""



    def make_move(self, board, test=False):
        moves = [move for move in board.legal_moves]
        rand_index = random.sample(range(len(moves)), 1)[0]
        return rand_index

