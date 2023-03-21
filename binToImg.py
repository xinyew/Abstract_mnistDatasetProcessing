import keras
from keras.datasets import mnist

import numpy as np
from PIL import Image, ImageOps
import os
import sys

import binToLabel

# Helper function to save images
def save_image(filename, data_array):
    im = Image.fromarray(data_array.astype('uint8'))
    im_invert = ImageOps.invert(im)
    im_invert.save(filename)

# Load MNIST Data
def loadMNIST():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


# Convert MNIST bin files to images with labels
def binToImg(target):
    x_train, y_train, x_test, y_test = loadMNIST()

    if target == "test":
        DIR_NAME = "testDataset"
        SIZE = 10000
        x = x_test
        y = y_test
    elif target == "train":
        DIR_NAME = "trainDataset"
        SIZE = 60000
        x = x_train
        y = y_train
    else:
        raise Exception("Target has to be test or train")
    
    if not os.path.exists(DIR_NAME):
        os.mkdir(DIR_NAME)

    for i in range(SIZE):
        pathName = "{0}/{1:01d}".format(DIR_NAME, y[i])
        if not os.path.exists(pathName):
            os.mkdir(pathName)
        filename = "{0}/{1}.jpg".format(pathName, i)
        print(filename)
        save_image(filename, x[i])

# Test function to verify the correctness of labels
def _test():
    for i in range(20):
        filename = str(i)
        print(filename)
        # print(y_test[i])
        print(binToLabel.getTestLabel(i, "test"))
        print()


binToImg(sys.argv[1])