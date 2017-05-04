##################################################
# EE 148 Assignment 4
#
# Author:   Andrew Kang
# File:     utility.py
# Desc:     Defines utility functions.
##################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Suppress compiler warnings.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


##############################
# PARAMETERS
##############################

# Directories
DATA_DIR = "../data/"
IMAGE_DIR = DATA_DIR + "images/"
RESULTS_DIR = "../results/"
VISUALIZATION_DIR = "../images/"
PREPROCESSED_DIR = DATA_DIR + "preprocessed/"

# Model parameters
INCEPTIONV3_SIZE = 299

# Model training parameters
OPTIMIZER = "rmsprop"
BATCH_SIZE = 32
EPOCHS = 10
VERBOSE = 1
SAVE = 1

##############################
# LOSS FUNCTIONS - 1x1
##############################

def F_loc(l, g):
    return tf.squared_difference(l, g) / 2

def F_conf(_, c):
    return - tf.log(c)

def F(l, g, c):
    return F_conf(c) + F_loc(l, g)


##############################
# IMAGE FUNCTIONS
##############################

def show_image(image):
    """
    Show given image.
    """

    plt.xticks([]), plt.yticks([])
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()


##############################
# VISUALIZATION FUNCTIONS
##############################

def visualize_cmatrix(model, X_test, Y_test, filename):
    """
    Visualize the confusion matrix for a model on a validation set.
    """

    # Predict on the test set.
    Y_true = to_multiclass(Y_test)
    Y_predict = to_multiclass(model.predict(X_test))

    # Calculate the confusion matrix.
    cmatrix = confusion_matrix(Y_true, Y_predict)
    # show_image(cmatrix)

    # Save figure.
    # plt.savefig(VISUALIZATION_DIR + filename)
    return cmatrix
