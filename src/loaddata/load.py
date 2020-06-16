import numpy as np
import sys

from keras.datasets import mnist
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
# import cPickle
import gzip
import matplotlib.pyplot as plt

# load train and test dataset
def load_mnist_online():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    # trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    # testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    # trainY = to_categorical(trainY)
    # testY = to_categorical(testY)
    return trainX, trainY, testX, testY

def load_mnist_offline():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    training_data, validation_data, test_data = cPickle.load(f)
    """
    f = gzip.open('../data/MNIST/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_mnist():
    print('Loading MNIST database, this might take a while...')
    data = tfds.as_numpy(tfds.load(
        'mnist',
        batch_size=-1,
        as_supervised=True,
    ))

    training_data = list(data['train'])
    testing_data = list(data['test'])
    training_data[0] = training_data[0][:, :, :, 0]
    testing_data[0] = testing_data[0][:, :, :, 0]
    training_data = tuple(training_data)
    testing_data = tuple(testing_data)

    return (training_data, testing_data)

def load_emnist():
    print('Loading EMNIST database, this might take a while...')
    data = tfds.as_numpy(tfds.load(
        'emnist',
        batch_size=-1,
        as_supervised=True,
    ))

    training_data = list(data['train'])
    testing_data = list(data['test'])
    training_data[0] = tf.image.rot90(training_data[0], k=3)
    testing_data[0] = tf.image.rot90(testing_data[0], k=3)
    training_data[0] = tf.image.flip_left_right(training_data[0])
    testing_data[0] = tf.image.flip_left_right(testing_data[0])
    testing_data[0] = testing_data[0].numpy()
    training_data[0] = training_data[0].numpy()
    training_data[0] = training_data[0][:, :, :, 0]
    testing_data[0] = testing_data[0][:, :, :, 0]
    training_data = tuple(training_data)
    testing_data = tuple(testing_data)

    return (training_data, testing_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


#mnist_train, mnist_test = load_mnist()

#emnist_train, emnist_test = load_emnist()

#trainX, trainY, testX, testY = load_mnist_online()

#training_data, validation_data, test_data = load_mnist_offline()












