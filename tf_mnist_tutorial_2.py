"""
TensorFlow Tutorial from https://www.tensorflow.org/tutorials/mnist/pros/
"""

from tensorflow.examples.tutorials.mnist import input_data

def main():
    """Main function"""

    print("Loading MNIST data")
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print("Data loaded")


if __name__ == "__main__":
    main()
