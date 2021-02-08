import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def load_emnist(split = 'balanced'):
    """ Loads the EMNIST dataset via tensorflow_datasets.load(). At first execution downloads the database to a local
    directory (see documentation for tensorflow_datasets.load()), after that grabs database from this local directory.
    Returns training_data and testing_data each as a tuple of an input array training_sizex28x28 resp.
    test_sizex28x28 with values between 0 and 255 and a result array of size training_size resp. test_size containing
    the associated labels as integers.
    Warning: these tuples are very large and take up a lot of RAM
    Possible splits:
    'balanced' (default): 47 classes, 112800 training samples, 11800 test samples
    'letters' : 26 classes, 88800 training samples, 14800 test samples
    'mnist' : 10 classes, 60.000 training samples, 10.000 test samples
    """

    print('Loading EMNIST database, this might take a while...')
    data = tfds.as_numpy(tfds.load(
        "emnist/" + split,
        batch_size=-1,
        as_supervised=True,
    ))

    # reshaping
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

    # correct error in dataset letters
    if split == 'letters':
        testing_data[1] = testing_data[1]-1
        training_data[1] = training_data[1] - 1

    training_data = tuple(training_data)
    testing_data = tuple(testing_data)

    return (training_data, testing_data)