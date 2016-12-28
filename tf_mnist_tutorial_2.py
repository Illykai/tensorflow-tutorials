"""
TensorFlow Tutorial from https://www.tensorflow.org/tutorials/mnist/pros/
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main():
    """Main function"""

    print("Loading MNIST data")
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print("Data loaded")

    sess = tf.InteractiveSession()

    # Data specification
    img_x = 28
    img_y = 28
    num_pixels = img_x * img_y
    num_classes = 10

    # Hyperparameters
    learning_rate = 0.5
    max_steps = 1000
    batch_size = 100

    # Inputs: pixel values x and class labels y_
    x = tf.placeholder(tf.float32, shape=[None, num_pixels])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

    # Model parameters: Weights W and biases b
    W = tf.Variable(tf.zeros([num_pixels, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

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

if __name__ == "__main__":
    main()
