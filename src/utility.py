##################################################
# EE 148 Assignment 4
#
# Author:   Andrew Kang
# File:     utility.py
# Desc:     Defines utility functions.
##################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
BATCH_SIZE = 128
EPOCHS = 8
VERBOSE = 1
SAVE = 1

##############################
# LOSS FUNCTIONS - 1x1
##############################

def F_loc(l, g):
    return tf.reduce_sum(tf.squared_difference(l, g)) / 2

def F_conf(c):
    return - tf.log(c)

def F(y_true, y_pred):
    return F_conf(y_pred[:, :, :, 4]) + F_loc(y_true[:, :, :, :4], y_pred[:, :, :, :4])


##############################
# PRIOR FUNCTIONS - 1x1
##############################

def transform(Y):
    """
    Transforms the dataset with respect to priors.
    """

    Y[:, :, :, 2:4] -= 1
    
    return Y

def expand_box(box):
    """
    Transform the given bounding box to 
    """

    box[2] += 1
    box[3] += 1
    box = [int(round(elem * INCEPTIONV3_SIZE)) for elem in box]

    return box


##############################
# IMAGE FUNCTIONS
##############################

def show_image(image, show=True):
    """
    Show given image.
    """

    # plt.xticks([]), plt.yticks([])
    plt.imshow(image) #, cmap=plt.get_cmap('gray'))

    if show:
        plt.show()

def show_box(image, box):
    """
    Show given image with the given overlaying bounding box.
    """

    show_image(image, show=False)
    
    if box[0] > box[2] or box[1] > box[3]:
        return

    # Overlay box.
    box = expand_box(box)
    rect = patches.Rectangle((box[0], box[3]), box[2] - box[0], box[3] - box[1],
                             linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)

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
