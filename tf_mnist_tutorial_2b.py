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

    # Data specification
    img_x = 28
    img_y = 28
    img_color_channels = 1 # greyscale images
    num_pixels = img_x * img_y
    num_classes = 10

    # Hyperparameters
    learning_rate = 0.5
    max_steps = 1000
    batch_size = 100
    field_size_conv1 = 5 # size of receptive field
    features_conv1 = 32 # output features for layer 1

    # Inputs: pixel values x and class labels y_
    x = tf.placeholder(tf.float32, shape=[None, num_pixels])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

    # First convolutional layer
    W_conv1 = weight_variable([field_size_conv1, field_size_conv1, img_color_channels, features_conv1])
    b_conv1 = bias_variable([features_conv1])

    # Wrangling x into the correct dimensions (i.e. a 28x28 image)
    x_image = tf.reshape(x, [-1, img_x, img_y, img_color_channels])

    # Output scores as "unnormalized log probabilities"
    y = tf.matmul(x, W) + b
    # Cross-entropy loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # Optimization step
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


    # Do the optimization
    sess.run(tf.global_variables_initializer())
    for i in range(max_steps):
        # Grab the next batch of training examples
        batch = mnist.train.next_batch(batch_size)
        # Bind the placeholders we defined to the batch data loaded and do a step
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    
    # Model is considered correct if it gives true class the highest probability
    # Note that this is actually represented as unnormalized log probability
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # Convert from bools to floats and take the mean to get the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels})

    print("Final accuracy: {0}".format(result))

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
