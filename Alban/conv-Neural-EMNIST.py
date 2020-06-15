from scipy import io as sio
import tensorflow as tf
import numpy as np
import keras

from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#step 1 loading and preprocessing EMNIST dataset 

mat = sio.loadmat('emnist-letters.mat')
data = mat['dataset']



#x_train, x_test, y_train, y_test = train_test_split(data['train'][0,0]['images'][0,0],
                                                   # data['train'][0,0]['images'][0,0],
                                                   # test_size=0.2)
                                                    

#                                                              random_state=42

x_train = data['train'][0,0]['images'][0,0] 
y_train = data['train'][0,0]['labels'][0,0]
x_test = data['test'][0,0]['images'][0,0]
y_test = data['test'][0,0]['labels'][0,0]

#_train = data['test'][0,0]['labels'][0,0]
#using cros validation 
#al_start = x_train.shape[0] - x_test.shape[0]
#_val =x_train[val_start:x_train.shape[0]]
#_val = y_train[val_start:y_train.shape[0]]
#_train = x_train[0:val_start]
#_train = y_train[0:val_start]

#reshape the arrays into image

              

x_train = x_train.reshape( (x_train.shape[0], 28, 28), order='F')


    
    
    
  
    
#_train.shape, y_train.shape

#_test.shape, y_test.shape


#_train = x_train[..., tf.newaxis]
#_test = x_test[..., tf.newaxis]

#_train.shape

#_test.shape

#p.min(x_train), np.max(x_train)

#x_train = x_train / 255.
#x_test=x_test/255

#p.min(x_train), np.max(x_train)
#x_train = x_train.reshape( (x_train.shape[0], 28, 28), order='F')

#y_train = y_train.reshape( (y_train.shape[0], 28, 28), order='F')'''

x_test = x_test.reshape( (x_test.shape[0], 28, 28), order='F')


#number of unique classes to be used
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)




print(x_train.shape)
print('test shape', x_test.shape)...

""x_train = x_train.reshape(-1, 28,28, 1)
x_test = x_test.reshape(-1, 28,28, 1)
x_train.shape, x_test.shape

#converting the data from int8 to float32

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_tesft / 255.f


#create model

batch_size = 64
epochs = 7
num_classes = 26

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes+1, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

#train
model_train = model.fit(x_train, y_train_one_hot, batch_size=batch_size,epochs=epochs, verbose =1)

#test evaluation
test_eval = model.evaluate(x_test, y_test_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1]

accuracy = model_train.history['acc']
val_accuracy = model_train.history['val_acc']
loss = model_train.history['loss']
val_loss = model_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#testing the working pace of the model to predict
