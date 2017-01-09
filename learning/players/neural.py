"""
Players that use neural networks to choose actions
"""

import collections
import csv
import enum
import math
import random
import numpy
import tensorflow as tf
from learning.players.basic import Player

DataSets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
WEIGHT_STD_DEV = 0.1
BIAS_DEFAULT = 0.1
WEIGHT_DECAY = 0.0025
# WEIGHT_DECAY = 0.005

def main():
    """Handy entry point for testing"""
    test_percent = 0.2
    train_percent = 0.6
    label_type = LabelType.winner
    # label_type = LabelType.action
    filename = "data/2016_12_28_21_25_00_tic_tac_toe_random_vs_random_games_1000.csv"
    # filename = "data/2017_01_04_09_02_51_games_10000__tic_tac_toe_negamax_vs_random_games.csv"
    data_sets = read_data_sets(filename, test_percent, train_percent, label_type)
    restore_filename = None
    # restore_filename = "data/random_vs_random.ckpt"
    # restore_filename = "data/negamax_vs_random.ckpt"
    save_filename = None
    # save_filename = "data/random_vs_random.ckpt"
    # save_filename = "data/negamax_vs_random.ckpt"
    train_neural_net(data_sets, restore_filename, save_filename)

def generate_win_prediction_player_from_dummy():
    """Make a dummy player for testing"""
    dummy_net = DummyNetwork()
    return WinPredictionPlayer(dummy_net)

class WinPredictionPlayer(Player):
    """
    Player that plays by looking ahead and picking the action their NN
    thinks will win
    """

    def __init__(self, network):
        self.network = network

    def get_action(self, game):
        """
        Get the player's move given the current game state
        """

        valid_actions = game.get_valid_moves()
        current_state = game.get_state()
        successors = [(game.get_state_successor(current_state, action), action)
                      for action in valid_actions]
        player = game.get_active_player()
        best_prob = -1.0
        best_actions = []
        for (state, action) in successors:
            win_probs = self.network.query(state)
            if win_probs[player] > best_prob:
                best_prob = win_probs[player]
                best_actions = [action]
            elif win_probs[player] == best_prob:
                best_actions.append(action)
        return random.choice(best_actions)

class DummyNetwork:
    """"
    Just a stub network for testing
    """

    def query(self, state):
        """Dummy query for testing"""
        if str(state) == str([0, 1, 0, 0, 0, 0, 0, 0, 0]):
            return [0.1, 0.7, 0.2]
        else:
            return [0.1, 0.3, 0.6]

class DataSet(object):
    """DataSet format based on TensorFlow MNIST example"""

    def __init__(self,
                 states,
                 labels):
        """Construct a DataSet."""
        assert states.shape[0] == labels.shape[0], (
            'states.shape: %s labels.shape: %s' % (states.shape, labels.shape))

        self.num_examples = states.shape[0]
        self.states = states
        self.labels = labels
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self.num_examples)
            numpy.random.shuffle(perm)
            self.states = self.states[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.states[start:end], self.labels[start:end]

class LabelType(enum.Enum):
    """LabelType enumeration"""
    winner = 0
    action = 1

class DataSplit(enum.Enum):
    """LabelType enumeration"""
    train = 0
    validation = 1
    test = 2

def read_data_sets(filename,
                   test_percent,
                   train_percent,
                   label_type):
    """Load data from the games"""

    # Generate parallel arrays of states and labels. We keep the states in 
    states = []
    labels = []
    with open(filename, "r") as data_file:
        reader = csv.reader(data_file, delimiter=",", quotechar='"')
        count = 0
        state_history = []
        winner = []
        for row in reader:
            if count == 0:
                state_history = row
            elif count == 1:
                action_history = row
            else:
                winner = int(row[0])
                one_hot_winner_array = numpy.zeros(3, dtype=numpy.uint8)
                one_hot_winner_array[winner] = 1
                # Insert a dummy action for the final move so every states
                # has an action associated with it
                action_history = [int(action) for action in action_history]
                action_history.append(9)
                # Label every state in the game
                for index, state in enumerate(state_history):
                    # Wrangle the state string into a numpy array
                    state_string = state[1:-1]
                    state_ints = state_string.split(", ")
                    state_ints = [int(num) for num in state_ints]
                    states.append(numpy.array(state_ints, dtype=numpy.uint8))

                    one_hot_action_array = numpy.zeros(10, dtype=numpy.uint8)
                    one_hot_action_array[action_history[index]] = 1

                    # We did a bit of work we didn't need to. Blah.
                    if label_type == LabelType.winner:
                        label_array = one_hot_winner_array
                    else:
                        label_array = one_hot_action_array

                    labels.append(label_array)

            count = (count + 1) % 3

    states_array = numpy.array(states)
    labels_array = numpy.array(labels)
    print("Data loaded from %s: " % filename)
    print("States array - Shape %s - Type %s "
          % (str(states_array.shape), str(states_array.dtype)))
    print("Labels array - Shape %s - Type %s "
          % (str(labels_array.shape), str(labels_array.dtype)))

    # Randomize the data for partitioning
    perm = numpy.arange(states_array.shape[0])
    numpy.random.shuffle(perm)
    states_array = states_array[perm]
    labels_array = labels_array[perm]

    # Compute the sizes of the partitions
    data_size = states_array.shape[0]
    train_size = int(math.floor(data_size * train_percent))
    test_size = int(math.floor(data_size * test_percent))
    validation_size = data_size - train_size - test_size

    # Make sure the partition sizes are valid
    assert test_size + train_size + validation_size == states_array.shape[0]

    # Do the partitioning
    test_states = states_array[:test_size]
    test_labels = labels_array[:test_size]
    validation_states = states_array[test_size:(test_size + validation_size)]
    validation_labels = labels_array[test_size:(test_size + validation_size)]
    train_states = states_array[-train_size:]
    train_labels = labels_array[-train_size:]

    # Convert the test and validation data into distributions
    test_states, test_labels = generate_state_label_distribution(test_states,
                                                                 test_labels)
    validation_states, validation_labels = generate_state_label_distribution(
        validation_states, validation_labels)

    print("Final data sizes:")
    print("Train:    %s" % str(train_states.shape))
    print("Test:     %s" % str(test_states.shape))
    print("Validate: %s" % str(validation_states.shape))

    train = DataSet(train_states, train_labels)
    validation = DataSet(validation_states, validation_labels)
    test = DataSet(test_states, test_labels)
    return DataSets(train=train, validation=validation, test=test)

def generate_state_label_distribution(states, labels):
    """Compute the distribution over labels for each state"""
    # Convert the test and validation data into distributions
    state_dict = {}

    # Build the un-normalized distributions
    index = 0
    for state in states:
        # Convert the state into a dictionary key
        state_string = str(state)
        label = labels[index]

        # Labels are 1-hot, so we can just sum them up
        if state_string not in state_dict:
            state_dict[state_string] = label
        else:
            total_labels = state_dict[state_string]
            total_labels += label
            state_dict[state_string] = total_labels
        index += 1

    # Normalize the distributions
    dist_states = []
    dist_labels = []
    for state_string, label_array in state_dict.items():
        # Strip the brackets off "[x y z]" and convert back to a numpy array
        state_string = state_string[1:-1]
        dist_states.append(numpy.fromstring(state_string, sep=" ", dtype=numpy.uint8))
        label_sum = label_array.sum()
        if label_sum > 1:
            label_array = label_array / label_sum
        dist_labels.append(label_array)

    return numpy.array(dist_states), numpy.array(dist_labels)


def train_neural_net(data, restore_filename=None, save_filename=None):
    """Main function"""
    sess = tf.InteractiveSession()

    ### Data specification
    img_x = 3
    img_y = 3
    img_color_channels = 1 # greyscale images
    num_pixels = img_x * img_y
    num_classes = 3
    # num_classes = 10
    
    ### Hyperparameters
    # Layer 1
    field_size_conv1 = 2
    num_features_conv1 = 64
    # Layer 2
    field_size_conv2 = 2
    num_features_conv2 = 64
    # Layer 3
    field_size_conv3 = 2
    num_features_conv3 = 64
    # Layer 4
    num_features_fc1 = 512
    # Training
    learning_rate = 1e-4 * 10
    # max_steps = 20000 * 2
    max_steps = 10000
    batch_size = 50

    ### Variable definitions
    # Inputs: pixel values x and class labels y_
    x = tf.placeholder(tf.float32, shape=[None, num_pixels], name="image")
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name="class")
    x_image = tf.reshape(x, [-1, img_x, img_y, img_color_channels],
                         name="reshape_to_square")

    ### Computation graph
    ## Layer 1 - Convolve -> ReLU
    # Wrangling x into the correct dimensions (i.e. a 28x28 image)
    with tf.name_scope("conv_layer_1"):
        h_conv1 = nn_layer_conv(x_image, img_color_channels, field_size_conv1,
                                num_features_conv1, "convolve")

    ## Layer 2 - Convolve -> relu
    with tf.name_scope("conv_layer_2"):
        h_conv2 = nn_layer_conv(h_conv1, num_features_conv1, field_size_conv2,
                                num_features_conv2, "convolve")

    ## Layer 3 - Convolve -> relu
    with tf.name_scope("conv_layer_3"):
        h_conv3 = nn_layer_conv(h_conv2, num_features_conv2, field_size_conv3,
                                num_features_conv3, "convolve")

    ## Layer 4 - Fully connected: Mutiply -> ReLU -> Dropout
    # We"ll have gone through 2 lots of max pooling, which will have shrunk our
    # dimensions and need to wrangle the output of the conv layers into the
    # correct shape
    input_dimension_fc1 = img_x * img_y * num_features_conv3
    h_conv3_flat = tf.reshape(h_conv3, [-1, input_dimension_fc1],
                              name="flatten_to_vector")
    h_fc = nn_layer_fc(h_conv3_flat, input_dimension_fc1, num_features_fc1,
                       "fc_layer_1")

    # Dropout for overfitting reduction
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar("dropout_keep_probability", keep_prob)
        dropped = tf.nn.dropout(h_fc, keep_prob)

    ## Layer 5 - Readout layer - Fully connected: Multiply only
    # Scores are unnormalize log probabilites
    y_conv = nn_layer_fc(dropped, num_features_fc1, num_classes, "fc_layer_2",
                         act=tf.identity)

    # Cross-entropy loss function
    with tf.name_scope("cross_entropy"):
        diff = tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)
        with tf.name_scope("total"):
            cross_entropy = tf.reduce_mean(diff)
            tf.add_to_collection("losses", cross_entropy)
    tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.name_scope("total_loss"):
        total_loss = tf.add_n(tf.get_collection("losses"), "total_loss")
    tf.summary.scalar("total_loss", total_loss)

    ## Optimization step
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(total_loss)

    ### Training

    # Merge summary statistics
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("data/train_ttt", sess.graph)
    test_writer = tf.summary.FileWriter("data/test_ttt")

    def feed_dict(batch_size, data_split, use_dropout):
        # Gonna use this to fill up the feed 
        if data_split == DataSplit.train:
            xs, ys = data.train.next_batch(batch_size)
        elif data_split == DataSplit.test:
            xs, ys = data.test.states, data.test.labels
        elif data_split == DataSplit.validation:
            xs, ys = data.validation.states, data.validation.labels
        else:
            # We messed up!
            pass
        k = 0.5 if use_dropout else 1.0
        result = {x: xs, y_: ys, keep_prob: k}
        return result

    # Do the optimization
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if restore_filename is not None:
        saver.restore(sess, restore_filename)
 
    for i in range(max_steps):
        # Grab the next batch of training examples
        if i % 100 == 0:
            feed = feed_dict(batch_size, DataSplit.validation, False)
            summary, entropy = sess.run([merged, cross_entropy], feed_dict=feed)
            test_writer.add_summary(summary, i)
            print("Step %d, validation cross entropy %g" % (i, entropy))
            feed = feed_dict(batch_size, DataSplit.train, False)
            entropy = sess.run(cross_entropy, feed_dict=feed)
            print("Step %d, training cross entropy %g" % (i, entropy))            
        else:
            if i % 100 == 99:
                # Write out summary statistics about the training
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                feed = feed_dict(batch_size, DataSplit.train, True)
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed,
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, "step%d" % i)
                train_writer.add_summary(summary, i)
            else:
                # Just do a regular step
                feed = feed_dict(batch_size, DataSplit.train, True)
                summary, _ = sess.run([merged, train_step], feed_dict=feed)
                train_writer.add_summary(summary, i)
        if i % 1000 == 0:
            # Save off a checkpoint
            if save_filename is not None:
                saver.save(sess, save_filename)

    # Test against the full test set
    feed = feed_dict(data.train.num_examples, DataSplit.test, False)
    summary, entropy = sess.run([merged, cross_entropy], feed_dict=feed)
    test_writer.add_summary(summary, max_steps)
    print("Final test cross entropy: %g" % entropy)

def bias_variable(shape):
    """
    Initialize a tensor of biases to 0.1
    """
    initial = tf.constant(BIAS_DEFAULT, shape=shape)
    var = tf.Variable(initial)
    weight_decay = tf.mul(tf.nn.l2_loss(var), WEIGHT_DECAY, name='weight_loss')
    tf.add_to_collection("losses", weight_decay)
    return var

def weight_variable(shape):
    """
    Initialize the values in a weight variable tensor with draws from a zero mean
    0.1 standard deviation Gaussian.
    """
    initial = tf.truncated_normal(shape, stddev=WEIGHT_STD_DEV)
    var = tf.Variable(initial)
    weight_decay = tf.mul(tf.nn.l2_loss(var), WEIGHT_DECAY, name='weight_loss')
    tf.add_to_collection("losses", weight_decay)
    return var

def conv2d(x, W):
    """
    Peform a 2d convolution with stride 1 and zero padding to make the output the size
    of the input.
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def variable_summaries(var):
    """
    Attach a summary statistics handle to a variable
    """
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)

def nn_layer_fc(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Helper for making a simple fully connected neural net layer

    Does matrix multiply, bias add, and relu. Also sets up handy name scoping.
    """
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope("biases"):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope("Wx_plus_b"):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram("pre_activations", preactivate)
        activations = act(preactivate, name="activation")
        tf.summary.histogram("activations", activations)
    return activations

def nn_layer_conv(input_tensor, input_dim, filter_dim, output_dim, layer_name, act=tf.nn.relu):
    """Helper for making a convolutional neural net layer

    Does convolution, bias add, relu, and max_pool. Also sets up handy name scoping.
    """
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            weights = weight_variable([filter_dim, filter_dim, input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope("biases"):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope("x_conv_W_plus_b"):
            preactivate = conv2d(input_tensor, weights) + biases
            tf.summary.histogram("pre_activations", preactivate)
        activations = act(preactivate, name="activation")
        tf.summary.histogram("activations", activations)

    return activations


if __name__ == "__main__":
    main()

