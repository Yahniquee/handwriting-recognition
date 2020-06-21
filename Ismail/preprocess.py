import os
import glob
import cv2
import numpy as np
import pandas as pd



def preprocess(img, output_shape=(130, 30)):
    (w_output, h_output) = output_shape   # w-wide and h-hight of image
    (h, w) = img.shape
    fx = w / w_output
    fy = h / h_output
    f = max(fx, fy)

    size = (max(min(w_output, int(w / f)), 1), max(min(h_output, int(h / f)), 1))
    img = cv2.resize(img, size)

    temp = np.ones([h_output, w_output]) * 255  # numpy.ones() - return a new array of given shape,filled with ones
    temp[0 : size[1], 0 : size[0]] = img
    img = cv2.transpose(temp) #

    (mean, std_dev) = cv2.meanStdDev(img)
    (mean, std_dev) = (mean[0][0], std_dev[0][0])
    img = img - mean
    img = img / std_dev if std_dev > 0 else img

    return img



