"""
TensorFlow Tutorial from https://www.tensorflow.org/tutorials/mnist/pros/
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

WEIGHT_STD_DEV = 0.1
BIAS_DEFAULT = 0.1

def main():
    """Main function"""
    print("Loading MNIST data")
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    print("Data loaded")

    sess = tf.InteractiveSession()

    ### Data specification
    img_x = 28
    img_y = 28
    img_color_channels = 1 # greyscale images
    num_pixels = img_x * img_y
    num_classes = 10

    ### Hyperparameters
    # Layer 1
    field_size_conv1 = 5
    num_features_conv1 = 32
    # Layer 2
    field_size_conv2 = 5
    num_features_conv2 = 64
    # Layer 3
    num_features_fc1 = 1024
    # Training
    learning_rate = 1e-4
    max_steps = 20000
    batch_size = 50

    ### Variable definitions
    # Inputs: pixel values x and class labels y_
    x = tf.placeholder(tf.float32, shape=[None, num_pixels], name="image")
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name="class")
    x_image = tf.reshape(x, [-1, img_x, img_y, img_color_channels],
                         name="reshape_to_square")

    ### Computation graph
    ## Layer 1 - Convolve -> ReLU -> MaxPool
    # Wrangling x into the correct dimensions (i.e. a 28x28 image)
    with tf.name_scope("conv_maxpool_layer_1"):
        h_conv1 = nn_layer_conv(x_image, img_color_channels, field_size_conv1,
                                num_features_conv1, "convolve")
        h_pool1 = nn_layer_max_pool_2x2(h_conv1, "max_pool")

    ## Layer 2 - Convolve -> ReLU -> MaxPool
    with tf.name_scope("conv_maxpool_layer_2"):
        h_conv2 = nn_layer_conv(h_pool1, num_features_conv1, field_size_conv2,
                                num_features_conv2, "convolve")
        h_pool2 = nn_layer_max_pool_2x2(h_conv2, "max_pool")

    ## Layer 3 - Fully connected: Mutiply -> ReLU -> Dropout
    # We"ll have gone through 2 lots of max pooling, which will have shrunk our
    # dimensions and need to wrangle the output of the conv layers into the
    # correct shape
    pooled_dims = int(img_x / 2 / 2)
    input_dimension_fc1 = pooled_dims * pooled_dims * num_features_conv2
    h_pool2_flat = tf.reshape(h_pool2, [-1, input_dimension_fc1],
                              name="flatten_to_vector")
    h_fc = nn_layer_fc(h_pool2_flat, input_dimension_fc1, num_features_fc1,
                       "fc_layer_1")

    # Dropout for overfitting reduction
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar("dropout_keep_probability", keep_prob)
        dropped = tf.nn.dropout(h_fc, keep_prob)

    ## Layer 4 - Readout layer - Fully connected: Multiply only
    # Scores are unnormalize log probabilites
    y_conv = nn_layer_fc(dropped, num_features_fc1, num_classes, "fc_layer_2",
                         act=tf.identity)

    # Cross-entropy loss function
    with tf.name_scope("cross_entropy"):
        diff = tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)
        with tf.name_scope("total"):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar("cross_entropy", cross_entropy)

    ## Optimization step
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        # Model is considered correct if it gives true class the highest probability
        # Note that this is actually represented as unnormalized log probability
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        # Convert from bools to floats and take the mean to get the accuracy
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    ### Training

    # Merge summary statistics
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("data/train", sess.graph)
    test_writer = tf.summary.FileWriter("data/test")

    def feed_dict(batch_size, train):
        # Gonna use this to fill up the feed 
        if train:
            xs, ys = mnist.train.next_batch(batch_size)
            k = 0.5
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        result = {x: xs, y_: ys, keep_prob: k}
        return result

    # Do the optimization
    sess.run(tf.global_variables_initializer())
    for i in range(max_steps):
        # Grab the next batch of training examples
        if i % 10 == 0:
            feed = feed_dict(batch_size, False)
            summary, acc = sess.run([merged, accuracy], feed_dict=feed)
            test_writer.add_summary(summary, i)
            print("Step %d, training accuracy %g" % (i, acc))
        # Bind the placeholders we defined to the batch data loaded and do a step
        else:
            if i % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                feed = feed_dict(batch_size, True)
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed,
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, "step%d" % i)
                train_writer.add_summary(summary, i)
            else:
                feed = feed_dict(batch_size, True)
                summary, _ = sess.run([merged, train_step], feed_dict=feed)
                train_writer.add_summary(summary, i)

    feed = feed_dict(batch_size, False)
    summary, acc = sess.run([merged, accuracy], feed_dict=feed)
    test_writer.add_summary(summary, max_steps)
    print("Final test accuracy: %g" % acc)

    print("Done")

def bias_variable(shape):
    """
    Initialize a tensor of biases to 0.1
    """
    initial = tf.constant(BIAS_DEFAULT, shape=shape)
    return tf.Variable(initial)

def weight_variable(shape):
    """
    Initialize the values in a weight variable tensor with draws from a zero mean
    0.1 standard deviation Gaussian.
    """
    initial = tf.truncated_normal(shape, stddev=WEIGHT_STD_DEV)
    return tf.Variable(initial)

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

def nn_layer_max_pool_2x2(x, layer_name):
    """
    Perform max pooling over blocks of x.
    """
    with tf.name_scope(layer_name):
        pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        tf.summary.histogram("pool activations", pool)
    return pool

if __name__ == "__main__":
    main()
