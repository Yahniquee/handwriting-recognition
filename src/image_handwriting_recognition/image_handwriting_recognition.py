import cv2 as cv
import numpy as np
import os as os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

def image2contourcoord(img, thresh_noise = 40, thresh_contarea = 80, thresh_binary = 200):
    img_gray = img.copy()
    img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)
    img_gray_inv = cv.subtract(255, img_gray)

    # denoising threshold for white background
    img_denoised = cv.threshold(img_gray_inv, thresh_noise, 255, cv.THRESH_TOZERO)[1]
    # invert image
    img_denoised = cv.bitwise_not(img_denoised, img_denoised)
    #cv.imshow("image_denoised", img_denoised)

    img_binary = cv.threshold(img_denoised, thresh_binary, 255, cv.THRESH_BINARY_INV)[1]

    # find contours
    cnts, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)

    img_rect = img.copy()
    area =[]
    cnts_good = []
    cnts_coord = []
    for c in cnts:
        area.append(cv.contourArea(c))
        if cv.contourArea(c) >= thresh_contarea:
            cnts_good.append(c)
            x,y,w,h = cv.boundingRect(c)
            cnts_coord.append([x,y,w,h])
            # only for visualization
            img_rect = cv.rectangle(img_rect, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print("Found %s contours that are treated as letters." % len(cnts_coord))
    return cnts_coord, img_denoised, img_binary, img_rect

def contourcoord2wordcoord(cnts_coord, space = 50):
    cnts_coord.sort(key = lambda x: x[0])
    words = [] # list of coords of letters corresponding to this word
    for i, coord in enumerate(cnts_coord):
        x, y, w, h = coord
        if i == 0:
            word = [coord]
        elif i == (len(cnts_coord)-1):
            word.append(coord)
            words.append(word)
        elif abs(cnts_coord[i+1][0] - (x+w)) <= space:
            word.append(coord)
        else:
            word.append(coord)
            words.append(word)
            word = []

    print("Seperated the image into %s words. " % len(words))
    return words

def wordcoords2wordimage(wordcoords, img_processed, img_plot):
    words = []
    words_cnts = []
    for wordcoord in wordcoords:
        letters = []
        img_word = img_plot.copy()
        for coord in wordcoord:
            x, y, w, h = coord
            letters.append(img_processed[y:(y + h), x:(x + w)])
            img_word = cv.rectangle(img_word, (x, y), (x + w, y + h), (0, 255, 0), 2)
        words.append(letters)
        words_cnts.append(img_word)

    return words, words_cnts

def wordimages2squares28(words):
    """Transforms words containing rectangle with letter from image processing to lists/words with 28x28 EMNIST-Type images """
    words_EMNIST = []
    for word in words:
        word_EMNIST = []
        for letter in word:
            h, w = np.shape(letter)
            size = np.max((h, w))
            square = np.zeros((size, size))
            test = size-w/2
            leftboundary = (size-w)/2
            topboundary = (size-h)/2
            square[int(topboundary):int((size)-topboundary+0.1), int(leftboundary):int((size)-leftboundary +0.1)] = letter
            square = cv.resize(square, dsize=(26,26), interpolation=cv.INTER_LINEAR)
            square = np.pad(square, pad_width=1, mode='constant', constant_values=0)
            square = square[:, :, np.newaxis]

            word_EMNIST.append(square)

        words_EMNIST.append(word_EMNIST)

    return words_EMNIST

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

def img2processedimg(img, h_smoothing):
    img_gray = img.copy()
    img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)
    img_processed = cv.fastNlMeansDenoising(img_gray, h=h_smoothing)
    img_processed = cv.bitwise_not(img_processed)
    img_processed = cv.threshold(img_processed, 10, 255, cv.THRESH_TOZERO)[1]
    img_processed = cv.threshold(img_processed, 100, 255, cv.THRESH_BINARY)[1]


    return img_processed

dic_balanced = np.array(["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","a","b","d","e","f","g","h","n","q","r","t",])
dic_letters = np.array(["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"])
dic_mnist = np.array(["0","1","2","3","4","5","6","7","8","9"])

parser = argparse.ArgumentParser(description='Handwriting Recognition from images using different neural networks    ')
parser.add_argument('-i', '--img', type=str, nargs='?', default='data/ML_at_eUHH.png', help='Filepath of input image (default data/ML_at_eUHH.png).')
parser.add_argument('-m', '--modeltype',type=str, nargs='?', default='balanced', help='Modeltype: balanced (default), letters or mnist.')
parser.add_argument('-p', '--plot',  action='store_true', default=False, help=' Decide whether to show plots, default is False.')
args = parser.parse_args()

imagepath = args.img
modeltype = args.modeltype

if modeltype == 'balanced':
    modelpath = "models/cnn_emnist_balanced_ep3.model"
    dic = dic_balanced
elif modeltype == 'letters':
    modelpath = "models/cnn_emnist_letters_ep3.model"
    dic = dic_letters
elif modeltype == "mnist":
    modelpath = "models/cnn_mnist_ep1.model"
    dic = dic_mnist
else:
    raise ValueError(modeltype + ' is not a valud modeltype. Choose balanced, letters or mnist')

img = cv.imread(imagepath)
if img is None:
    raise ValueError('Cannot find an image at ' + imagepath)

print("Imagepath: ", imagepath)
print("Modelpath: ", modelpath )
model = tf.keras.models.load_model(modelpath)



#cnn_emnist_balanced_ep3.model seems to work best, ep5 as well
# letters_3 is good as well

cnts_coord, img_denoised, img_binary, img_rect = image2contourcoord(img)
wordcoords = contourcoord2wordcoord(cnts_coord, space = 70)
img_processed = img2processedimg(img, 30)
words, words_cnts = wordcoords2wordimage(wordcoords, img_processed, img)
words_EMNIST = wordimages2squares28(words)

[words_str, words_class] = words_EMNIST2words_str(words_EMNIST, model, dic)


print("You may have written: ", " ".join(words_str))

plots = args.plot

if plots:
    fig, ax = plt.subplots(3, 3)
    if len(words_EMNIST[0]) <= 9:
        fig.suptitle("First word as input to NN", fontsize=20)
    else:
        fig.suptitle("First nine letters as input to NN", fontsize=20)
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
    cv.imshow("All contours that are treated as letters", img_rect)
    input("Press any key to exit.")


