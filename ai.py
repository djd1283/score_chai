"""Chess AI made in Tensorflow."""
import tensorflow as T
import random
import numpy as np

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

        piece_embs = T.get_variable('embs', shape=(num_pieces, self.piece_emb_size),
                                    initializer=T.contrib.layers.xavier_initializer(uniform=False))

        board_embs = T.nn.embedding_lookup(piece_embs, boards)

        conv1 = T.layers.conv2d(
            inputs=board_embs,
            filters=15,
            kernel_size=[2, 2],
            padding="same",
            activation=T.nn.tanh)

        conv2 = T.layers.conv2d(
            inputs=conv1,
            filters=15,
            kernel_size=[4, 4],
            padding="same",
            activation=T.nn.tanh)

        conv2_flat = T.contrib.layers.flatten(conv2)

        scores = build_linear_layer('linear', conv2_flat, 1, xavier=True)
        scores = T.reshape(scores, shape=(-1,), name='scores')
        probs = T.nn.sigmoid(scores)

        self.loss = T.reduce_mean(T.nn.sigmoid_cross_entropy_with_logits(logits=probs, labels=results))

        self.train = T.train.AdamOptimizer(self.lr).minimize(self.loss)

        # input each board position, and (for training) whether playing from that position resulted in a win
        self.i = {'boards': boards, 'results': results}
        # outputs the probability of winning from each board position, the highest probability board position,
        # and the loss function with respect to the label
        self.o = {'probs': probs, 'scores': scores, 'loss': self.loss}

    def make_move(self, boards, k=1.0, test=False):
        """Given boards corresponding to each move,
        returns the index of the board move the AI choses.

        boards - resulting boards after making every legal move
        k - temperature constant of softmax"""

        whites_turn = not boards[0].turn

        # we normalize by always telling the ai it is white's turn
        # by flipping the board. we then make the move as white.
        # this makes learning me from you easier for the model

        np_boards = convert_to_numpy(boards)

        scores = self.sess.run(self.o['scores'], feed_dict={self.i['boards']: np_boards})

        # sample from the possible choices - softmax
        probs = np.exp(k * scores) / np.sum(np.exp(k * scores))

        # black prefers the opposite distribution - minimize prob of white win
        if not whites_turn:
            probs = 1 - probs

        if test:
            choice = np.argmax(probs)
        else:
            choice = np.random.choice(range(len(boards)), p=probs)

        return choice

    def reinforce(self, boards, result):
        """Apply reinforcement learning signal to AI.

        boards - board positions seen throughout a single game of chess
        result - whether white won or lost

        Returns: loss function value"""

        np_boards = convert_to_numpy(boards)

        np_results = np.repeat(result, len(boards), axis=0)

        loss, _ = self.sess.run([self.loss, self.train],
                                feed_dict={self.i['results']: np_results, self.i['boards']: np_boards})


        return loss

    def save(self):
        assert self.save_dir is not None
        self.saver.save(self.sess, self.save_dir, global_step=self.step, write_meta_graph=False)
        self.step += 1


def convert_to_numpy(boards):
    """Converts chess boards to numpy format. Each
    chess board becomes 8x8 matrix of ids, each id
    corresponding to each piece in chess (6 pieces for
    white and 6 pieces for black).

    Returns: m x 8 x 8 numpy array for m boards.
    0 represents empty space, 1-6 represent
    (rook, knight, bishop, queen, king, pawn)
    for white and 7-12 for black."""
    num_boards = len(boards)
    np_boards = np.zeros((num_boards, 8, 8))
    for index in range(num_boards):
        map = boards[index].piece_map()

        # Fill in board with pieces
        for position in map:
            piece = str(map[position])
            y = position // 8
            x = position % 8

            np_boards[index, x, y] = piece_ids[piece]

    return np_boards


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
        initializer = T.random_normal_initializer(stddev=0.01)

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
    def make_move(self, boards, test=False):
        rand_index = random.sample(range(len(boards)), 1)[0]
        return rand_index

