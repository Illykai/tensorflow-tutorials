"""
TensorFlow Tutorial from https://www.tensorflow.org/tutorials/mnist/pros/
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

WEIGHT_STD_DEV = 0.1
BIAS_DEFAULT = 0.1
MAX_POOL_FIELD_SIZE = 2

def main():
    """Main function"""
    print("Loading MNIST data")
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print("Data loaded")

    sess = tf.InteractiveSession()

    ## Data specification
    img_x = 28
    img_y = 28
    img_color_channels = 1 # greyscale images
    num_pixels = img_x * img_y
    num_classes = 10

    ## Hyperparameters
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
    dropout_keep_prob = 0.5

    ## Variable definitions
    # Inputs: pixel values x and class labels y_
    x = tf.placeholder(tf.float32, shape=[None, num_pixels])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

    # Layer 1
    W_conv1 = weight_variable([field_size_conv1, field_size_conv1, img_color_channels, num_features_conv1])
    b_conv1 = bias_variable([num_features_conv1])

    # Layer 2
    W_conv2 = weight_variable([field_size_conv2, field_size_conv2, num_features_conv1, num_features_conv2])
    b_conv2 = bias_variable([num_features_conv2])

    # Layer 3
    # We'll have gone through 2 lots of max pooling, which will have shrunk our dimensions
    pooled_dims = int(img_x / 2 / 2)
    input_dimension_conv3 = pooled_dims * pooled_dims * num_features_conv2
    W_fc1 = weight_variable([input_dimension_conv3, num_features_fc1])
    b_fc1 = bias_variable([num_features_fc1])

    # Dropout probability vector (or rather its inverse)
    keep_prob = tf.placeholder(tf.float32)

    # Readout layer
    W_fc2 = weight_variable([num_features_fc1, num_classes])
    b_fc2 = bias_variable([num_classes])

    ## Computation graph definitions

    # Wrangling x into the correct dimensions (i.e. a 28x28 image)
    x_image = tf.reshape(x, [-1, img_x, img_y, img_color_channels])

    # Layer 1 - Convolve -> ReLU -> MaxPool
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Layer 2 - Convolve -> ReLU -> MaxPool
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Wrangling h_pool2 into the correct dimensions (a vector)
    h_pool2_flat = tf.reshape(h_pool2, [-1, input_dimension_conv3])

    # Layer 3 - Fully connected: Mutiply -> ReLU 
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout for overfitting reduction
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout layer - Scores are unnormalize log probabilites
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Cross-entropy loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    # Optimization step
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # Model is considered correct if it gives true class the highest probability
    # Note that this is actually represented as unnormalized log probability
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # Convert from bools to floats and take the mean to get the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Do the optimization
    sess.run(tf.global_variables_initializer())
    for i in range(max_steps):
        # Grab the next batch of training examples
        batch = mnist.train.next_batch(batch_size)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("Step %d, training accuracy %g" % (i, train_accuracy))
        # Bind the placeholders we defined to the batch data loaded and do a step
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: dropout_keep_prob})

    print("Final test accuracy: %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

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
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """
    Perform max pooling over blocks of x.
    """
    return tf.nn.max_pool(x, ksize=[1, MAX_POOL_FIELD_SIZE, MAX_POOL_FIELD_SIZE, 1], strides=[1, MAX_POOL_FIELD_SIZE, MAX_POOL_FIELD_SIZE, 1], padding='SAME')

if __name__ == "__main__":
    main()
