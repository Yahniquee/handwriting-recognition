import os as os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import loaddata.load as load
import matplotlib.pyplot as plt
import numpy as np

training_data, test_data = load.load_emnist("balanced")

(x_train, y_train) = training_data
(x_test, y_test) = test_data

x_train = x_train.reshape((x_train.shape[0], 28, 28), order='F')
x_test = x_test.reshape((x_test.shape[0], 28, 28), order='F')

classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# reshaping the training and testing data
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train.shape, x_test.shape

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

fig, ax = plt.subplots(3, 3)

dic_balanced = np.array(["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","a","b","d","e","f","g","h","n","q","r","t",])
dic_letters = np.array(["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"])
dic_mnist = np.array(["0","1","2","3","4","5","6","7","8","9"])

dic = dic_balanced

for i in range(0,9):
    img = x_train[i]
    classification = y_train[i]
    plt.subplot(3, 3, i + 1)
    plt.imshow(img.reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Class: " + dic[classification])
    plt.tight_layout()
    plt.axis('off')