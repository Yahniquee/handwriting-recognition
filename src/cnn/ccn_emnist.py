from scipy import io as sio
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# step 1 loading and preprocessing EMNIST dataset


import sys
import tensorflow_datasets as tfds


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
        'emnist/balanced',
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



training_data, test_data = load_emnist()
(x_train, y_train) = training_data
(x_test, y_test) = test_data

'''x_train, x_test, y_train, y_test = train_test_split(data['train'][0,0]['images'][0,0],
                                                    data['train'][0,0]['images'][0,0],
                                                    test_size=0.2, random_state=13)
'''


x_train = x_train.reshape((x_train.shape[0], 28, 28), order='F')
x_test = x_test.reshape((x_test.shape[0], 28, 28), order='F')

# number of unique classes
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# reshaping the training and testing data


print('train shape', x_train.shape)
print('test shape', x_test.shape)

# plots

plt.subplot(121)
plt.imshow(x_train[0, :, :], cmap='gray')
plt.title("Ground Truth : {}".format(y_train[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(x_test[0, :], cmap='gray')
plt.title("Ground Truth : {}".format(y_test[0]))

# reshaping the training and testing data
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train.shape, x_test.shape

# converting the data from int8 to float32

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.


from keras.utils import to_categorical

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Display the change for category label using one-hot encoding
print('Original label:', y_train[3])
print('After conversion to one-hot:', y_train_one_hot[0])


batch_size = 64
epochs = 3
num_classes = 47

# built the model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.summary()

'''Let's visualize the layers that you created in the above step by using the summary function. 
This will show some parameters (weights and biases) in each layer and also the total parameters in your model'''

# train the model/fiting
'''
model_train = model.fit(x_train, y_train_one_hot, batch_size=batch_size, epochs=epochs, verbose=1)

# In[1]:


test_eval = model.evaluate(x_test, y_test_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# In[22]:


# now we analyse our model and see what can be done to make it better

accuracy = model_train.history['accuracy']
# val_acc = model_train.history['val_accuracy']
loss = model_train.history['loss']
# val_loss = model_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
# plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# In[34]:


model.save("emnist_balanced_let_read.model")

# In[11]:

'''
new_model = tf.keras.models.load_model("emnist_balanced_let_read.model")
model = new_model

# In[12]:


prediktions =model.predict(x_test)
#print(prediktions)

# In[13]:


#print(np.argmax(prediktions[24]))

# In[23]:


predicted_classes = model.predict(x_test)
# need to use argmax function to find the one prediction with the highest probability
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

# In[24]:


predicted_classes.shape, y_test.shape



correct = np.where(22 == y_test)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[correct].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    plt.tight_layout()


# now for the incorrect ones
incorrect = np.where(predicted_classes[86] != y_test)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
    plt.tight_layout()

# In[30]:


'''Classification report will help us in identifying the misclassified classes in more detail.
You will be able to observe for which class the model performed bad out of the given 26 classes'''

from sklearn.metrics import classification_report

target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))

# In[ ]:




