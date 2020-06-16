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
    """Do not use this one"""
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
    """
    Taken from Nielsen, adapted for Python 3.
    Return the MNIST data as a tuple containing the training data,
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
    """ Loads the MNIST dataset via tensorflow_datasets.load(). At first execution downloads the database to a local
    directory (see documentation for tensorflow_datasets.load()), after that grabs database from this local directory.
    Returns training_data (60.000) and testing_data (10.000) each as a tuple of an input array 60.000x28x28 resp.
    10.000x28x28 with values between 0 and 255 and a result array of size 60.000 resp. 10.000 containing the associated
    label between 0 and 61 representing number, small letters and capital letters"""

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
    """ Loads the EMNIST dataset via tensorflow_datasets.load(). At first execution downloads the database to a local
    directory (see documentation for tensorflow_datasets.load()), after that grabs database from this local directory.
    Returns training_data (697932) and testing_data (116323) each as a tuple of an input array 697932x28x28 resp.
    116323x28x28 with values between 0 and 255 and a result array of size 60.000 resp. 10.000 containing the associated
    labels between 0
    Warning: these tuples are very large and take up a lot of RAM"""
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


# The following is only relevant for neuralnetnp by Nielsen.
def vectorized_result(j):
    """ From Nielsen, used in load_data_wrapper()
    Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data_wrapper():
    """ From Nielsen, adapted for Python 3.

    Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""

    tr_d, va_d, te_d = load_mnist_offline()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)










