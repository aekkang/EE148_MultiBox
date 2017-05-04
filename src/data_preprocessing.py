##################################################
# EE 148 Assignment 4
#
# Author:   Andrew Kang
# File:     data_preprocessing.py
# Desc:     Defines functions for preprocessing
#           the Caltech-UCSD Birds-200 dataset.
##################################################

import math
import matplotlib.pyplot as plt
import numpy as np
import cv2

from utility import *


##############################
# DATA PREPROCESSING
##############################

def save_data():
    """
    Save the Caltech-UCSD Birds-200 dataset.
    """

    # Load relevant files.
    image_paths = np.genfromtxt(DATA_DIR + "images.txt", dtype=None)
    train_test_split = np.genfromtxt(DATA_DIR + "train_test_split.txt", dtype=None)
    bounding_boxes = np.genfromtxt(DATA_DIR + "bounding_boxes.txt", dtype=None)

    X_train, Y_train, X_test, Y_test = [], [], [], []

    # Load and modify images.
    for i, (image_id, image_path) in enumerate(image_paths):
        if i % 100 == 0:
            print("Saving image: " + str(i))

        # Extract information.
        train_test = train_test_split[i][1]

        # Read and resize the image to the input size required by the network.
        image_path = image_path.decode("UTF-8")
        image = cv2.imread(IMAGE_DIR + image_path)
        
        # Values used to recalculate scaled bounding box.
        h, w, n_channels = image.shape
        image, vpadding, hpadding, scale = resize_to_square(image)
        hpadding /= scale
        vpadding /= scale
        side = float(max(w, h))

        # Recalculate bonuding box.
        box = bounding_boxes[i]
        x, y = box[1] + math.floor(hpadding), box[2] + math.floor(vpadding)
        dx, dy = box[3], box[4]
        
        box = (x / side, y / side, (x + dx) / side, (y + dy) / side)

        # Add the image and label to the datasets.
        if train_test:
            X_train.append(image)
            Y_train.append(box)
        else:
            X_test.append(image)
            Y_test.append(box)

    np.save(PREPROCESSED_DIR + "X_train", X_train)
    np.save(PREPROCESSED_DIR + "Y_train", Y_train)
    np.save(PREPROCESSED_DIR + "X_test", X_test)
    np.save(PREPROCESSED_DIR + "Y_test", Y_test)

def load_data():
    """
    Load and return the Caltech-UCSD Birds-200 dataset.
    """

    X_train = np.load(PREPROCESSED_DIR + "X_train.npy")
    Y_train = np.load(PREPROCESSED_DIR + "Y_train.npy")
    X_test = np.load(PREPROCESSED_DIR + "X_test.npy")
    Y_test = np.load(PREPROCESSED_DIR + "Y_test.npy")

    return (X_train, Y_train), (X_test, Y_test)

def resize_to_square(image):
    """
    Resize the given image to a 299x299 square, corresponding to an
    input for InceptionV3.
    """

    # Determine new dimensions.
    h, w, n_channels = image.shape
    scale = INCEPTIONV3_SIZE / float(max(w, h))
    new_dim = (int(scale * w), int(scale * h))
    
    # Determine padding.
    vpadding = (INCEPTIONV3_SIZE - new_dim[1]) / 2.
    hpadding = (INCEPTIONV3_SIZE - new_dim[0]) / 2.

    # Resize and pad image.
    image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(image,
                               math.floor(vpadding), math.ceil(vpadding),
                               math.floor(hpadding), math.ceil(hpadding),
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return image, vpadding, hpadding, scale


if __name__ == "__main__":
    save_data()
    (X_train, Y_train), (X_test, Y_test) = load_data()
