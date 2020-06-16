import numpy as np

def data_wrapper(training_data, test_data):
    """
    Tranforms our database as given by load.load_mnist() to Nielsen's form.

    Return a tuple containing ``(training_data, test_data)``. in a format more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 60,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``test_data`` is a list containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, te_d = list(training_data), list(test_data)
    tr_d[0] = tr_d[0].reshape(tr_d[0].shape[0], 784)
    te_d[0] = te_d[0].reshape(te_d[0].shape[0], 784)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_inputs = [x / 256 for x in training_inputs]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_inputs = [x / 256 for x in test_inputs]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e