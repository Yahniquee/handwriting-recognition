import cv2 as cv
import numpy as np

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

    print("Separated the image into %s words. " % len(words))
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
            square = square / 255.

            word_EMNIST.append(square)

        words_EMNIST.append(word_EMNIST)

    return words_EMNIST

def img2processedimg(img, h_smoothing):
    img_gray = img.copy()
    img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)
    img_processed = cv.fastNlMeansDenoising(img_gray, h=h_smoothing)
    img_processed = cv.bitwise_not(img_processed)
    img_processed = cv.threshold(img_processed, 10, 255, cv.THRESH_TOZERO)[1]
    img_processed = cv.threshold(img_processed, 100, 255, cv.THRESH_BINARY)[1]


    return img_processed