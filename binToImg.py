import keras
from keras.datasets import mnist

import numpy as np
from PIL import Image, ImageOps
import os

import binToLabel

def save_image(filename, data_array):
    im = Image.fromarray(data_array.astype('uint8'))
    im_invert = ImageOps.invert(im)
    im_invert.save(filename)

# Load MNIST Data
def loadMNIST():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test

# DIR_NAME = "JPEGImages"
# if os.path.exists(DIR_NAME) == False:
#     os.mkdir(DIR_NAME)

# # Save Images
# i = 0
# for li in [x_train, x_test]:
#     print("[---------------------------------------------------------------]")
#     for x in li:
#         filename = "{0}/{1:05d}.jpg".format(DIR_NAME,i)
#         print(filename)
#         save_image(filename, x)
#         i += 1

def binToImg(target, x, y):
    if target == "test":
        DIR_NAME = "testDataset"
        SIZE = 10000
    elif target == "train":
        DIR_NAME = "trainDataset"
        SIZE = 60000
    else:
        raise Exception("Target has to be test or train")
    
    for i in range(SIZE):
        pathName = "{0}/{1:01d}.jpg".format(DIR_NAME, y[i])
        if not os.path.exists(pathName):
            os.mkdir(pathName)
        filename = "{0}/{1}.jpg".format(pathName, i)
        print(filename)
        save_image(filename, x[i])

def _test():
    for i in range(20):
        filename = str(i)
        print(filename)
        # print(y_test[i])
        print(binToLabel.getTestLabel(i, "test"))
        print()

x_train, y_train, x_test, y_test = loadMNIST()
binToImg("test", x_test, y_test)