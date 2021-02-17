import numpy as np
import os as os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress warnings during tensorflow import
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from imageprocessing import *

def words_EMNIST2words_str(words_EMNIST, model, dic):
    words = [[], []]
    for word in words_EMNIST:
        word_str = str()
        word_class = []
        for letter in word:
            modelinput = np.array([letter])
            prediction = model.predict(modelinput)
            predicted_class = np.argmax(np.round(prediction))
            letter = dic[predicted_class]

            word_str+=str(letter)
            word_class.append(predicted_class)
        words[0].append(word_str)
        words[1].append(word_class)

    return words

# parse input
parser = argparse.ArgumentParser(description='Handwriting Recognition from images using CNNs.')
parser.add_argument('-i', '--img', type=str, nargs='?', default='data/ML.png',
                    help='Filepath of input image (default data/ML.png).')
parser.add_argument('-m', '--modeltype',type=str, nargs='?', default='balanced',
                    help='Modeltype: balanced (default), letters or mnist.')
parser.add_argument('-p', '--plot',  action='store_true', default=False,
                    help='Decide whether to show plots, default is False.')
args = parser.parse_args()

imagepath = args.img
modeltype = args.modeltype
plots = args.plot

dic_balanced = np.array(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M',
                       'N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','d','e','f','g','h','n','q','r','t'])
dic_letters = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X',
                     'Y','Z'])
dic_mnist = np.array(['0','1','2','3','4','5','6','7','8','9'])

if modeltype == 'balanced':
    modelpath = 'models/cnn_emnist_balanced_ep3.model'
    dic = dic_balanced
elif modeltype == 'letters':
    modelpath = 'models/cnn_emnist_letters_ep3.model'
    dic = dic_letters
elif modeltype == 'mnist':
    modelpath = 'models/cnn_mnist_ep1.model'
    dic = dic_mnist
else:
    raise ValueError(modeltype + ' is not a valid modeltype. Choose balanced, letters or mnist')

img = cv.imread(imagepath)
if img is None:
    raise ValueError('Cannot find an image at ' + imagepath)

print('Imagepath: ', imagepath)
print('Modelpath: ', modelpath)
model = tf.keras.models.load_model(modelpath)

# image processing
cnts_coord, img_denoised, img_binary, img_rect = image2contourcoord(img)
wordcoords = contourcoord2wordcoord(cnts_coord, space = 70)
img_processed = img2processedimg(img, 30)
words, words_cnts = wordcoords2wordimage(wordcoords, img_processed, img)
words_EMNIST = wordimages2squares28(words)

# model prediction
[words_str, words_class] = words_EMNIST2words_str(words_EMNIST, model, dic)

print('You may have written: ', ' '.join(words_str))

if plots:
    fig, ax = plt.subplots(3, 3)
    if len(words_EMNIST[0]) <= 9:
        fig.suptitle('First word as input to NN', fontsize=20)
    else:
        fig.suptitle('First nine letters as input to NN', fontsize=20)
    i = 0
    for j, letter in enumerate(words_EMNIST[0][:9]):
        if j % 3 == 0 and j != 0:
            i = i+1
        j = j - (i * 3)
        im = letter.reshape(28, 28)
        ax[i][j].imshow(im, cmap='gray', interpolation='none', aspect='equal')
        ax[i][j].axis('off')
        ax[i][j].set_title('Classification: ' + words_str[0][j + (i*3)])
    for row in ax:
        for cell in row:
            if not cell.images:
                cell.set_axis_off()

    fig.show()
    cv.imshow('All contours that are treated as letters', img_rect)
    input('Press any key to exit.')