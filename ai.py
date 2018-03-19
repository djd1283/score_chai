"""Chess AI made in Tensorflow."""
import tensorflow as tf
import random
import numpy as np

piece_ids = {'r': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5, 'p': 6,
             'R': 7, 'N': 8, 'B': 9, 'Q': 10, 'K': 11, 'P': 12, ' ': 0}

class ChessAI:
    def __init__(self, piece_emb_size, lr, **kwargs):
        self.piece_emb_size = piece_emb_size

        self.lr = lr  # learning rate
        self.build()

        # Create session and model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession(config=self.config)
        self.sess.run(self.init)
        self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)

    def build(self):
        """Build convolutional neural network."""

        num_pieces = 13  # 6 piece types * 2 sides + 1 for empty square

        boards = tf.placeholder(tf.int32, shape=(None, 8, 8), name='boards')
        reward = tf.placeholder(tf.float32, shape=(), name='reward')
        train_choice = tf.placeholder(tf.int32, shape=(), name='train_choice')

        piece_embs = tf.get_variable('embs', shape=(num_pieces, self.piece_emb_size),
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        board_embs = tf.nn.embedding_lookup(piece_embs, boards)

        conv1 = tf.layers.conv2d(
            inputs=board_embs,
            filters=32,
            kernel_size=[2, 2],
            padding="same",
            activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=32,
            kernel_size=[4, 4],
            padding="same",
            activation=tf.nn.relu)

        conv2_flat = tf.contrib.layers.flatten(conv2)

        scores = build_linear_layer('linear', conv2_flat, 1, xavier=True)
        scores = tf.reshape(scores, shape=(-1,), name='scores')
        print(scores.shape)
        print(train_choice.shape)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores,
                                                                       labels=train_choice)

        self.loss = cross_entropy * reward  # should this be negative?

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        probs = tf.nn.softmax(scores)

        choice = tf.argmax(probs)

        self.i = {'boards': boards, 'reward': reward, 'choice': train_choice}

        self.o = {'choice': choice, 'probs': probs, 'scores': scores, 'loss': self.loss}


    def make_move(self, boards):
        """Given boards corresponding to each move,
        returns the index of the board move the AI choses."""
        rand_index = random.sample(range(len(boards)), 1)[0]

        num_boards = len(boards)

        whites_turn = not boards[0].turn

        # we normalize by always telling the ai it is white's turn
        # by flipping the board. we then make the move as white.
        # this makes learning me from you easier for the model
        if not whites_turn:
            boards = [board.mirror() for board in boards]

        np_boards = convert_to_numpy(boards)

        probs = self.sess.run(self.o['probs'], feed_dict={self.i['boards']: np_boards})

        # sample from the possible choices
        choice = np.random.choice(range(num_boards), p=probs)

        #print('%s/%s' % (choice, num_boards))
        #print('turn: %s' % whites_turn)

        return choice

    def reinforce(self, examples, reward):
        """Apply reinforcement learning signal to AI"""

        # train on each example!
        for example in examples:

            boards = example[0]
            choice = example[1]

            np_boards = convert_to_numpy(boards)

            self.sess.run(self.train_op, feed_dict={self.i['choice']: choice,
                                                    self.i['boards']: np_boards,
                                                    self.i['reward']: reward})




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
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    else:
        initializer = tf.random_normal_initializer(stddev=0.01)

    input_size = input_tensor.get_shape()[-1]  #tf.shape(input_tensor)[1]
    with tf.variable_scope(name):
        scale_w = tf.get_variable('w', shape=(input_size, output_size),
                                  initializer= initializer) # tf.contrib.layers.xavier_initializer(uniform=False)) #

        scale_b = tf.get_variable('b', shape=(output_size,), initializer=tf.zeros_initializer())

    return tf.matmul(input_tensor, scale_w) + scale_b
