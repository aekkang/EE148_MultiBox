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
# PARTS_DIR = DATA_DIR + "parts/"
RESULTS_DIR = "../results/"
# VISUALIZATION_DIR = "../images/"
PREPROCESSED_DIR = DATA_DIR + "preprocessed/"

# Model parameters
INCEPTIONV3_SIZE = 299
# WARP_SIZE = 500
# N_CLASSES = 200

# Model training parameters
OPTIMIZER = "rmsprop"
BATCH_SIZE = 32
EPOCHS = 10
VERBOSE = 1
SAVE = 1

# # Head pose parameters.
# N_PARTS = 15
# HEAD_PARTS = [10, 1, 4, 5, 9, 14]
# LEFT_EYE = 6
# RIGHT_EYE = 10
# REF_ID = 6

# # Miscellaneous parameters.
# MODE = (0, 0, 1)
# MODE_KEYS = ["", "cropped_", "warped_"]
# N_CORRECTED = 5

# for i, e in enumerate(MODE):
#     if e:
#         MODE_KEY = MODE_KEYS[i]


##############################
# LOSS FUNCTIONS - 1x1
##############################

def F_loc(l, g):
    return tf.squared_difference(l, g) / 2
    return np.linalg.norm(l - g) ** 2 / 2

def F_conf(_, c):
    return - tf.log(c)

def F(l, g, c):
    return F_conf(c) + F_loc(l, g)

def loss(y_true, y_pred):
    print(y_pred.shape)
    return tf.squared_difference(y_true, y_pred) / 2
    # print(y_true.shape)
    # print(y_pred)
    # for i in range(len(y_true)):
    #     box_true = y_true[i, 0, 0, :]
    #     box_pred = y_pred[i, 0, 0, :]

    #     g = box_true
    #     l = box_pred[0]
    #     c = box_pred[1]

    return F(l, g, c)


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

def plot_accuracy(history, filename):
    """
    Plot the training and validation accuracy over the
    course of training a model.
    """

    # Load accuracy.
    history = history.item()
    training = history[b"acc"]
    validation = history[b"val_acc"]

    # Plot accuracy.
    plt.plot(training, label="Training accuracy")
    plt.plot(validation, label="Validation accuracy")
    
    # Set plot details.
    plt.title("Training and Validation Accuracy During Model Training")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')

    plt.xlim(0, EPOCHS)
    plt.ylim(0.0, 1.0)

    # Save figure.
    plt.savefig(VISUALIZATION_DIR + filename)

def to_multiclass(lst):
    """
    Convert a one-hot encoded array to multiclass.
    """
    
    return np.argmax(lst, axis=1)

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

def show_corrected_images(model, better_model, X_test, X_better_test, Y_test):
    """
    Show images on which a model classified incorrectly but
    a better model classified correctly.
    """

    # Predict on the test set.
    Y_true = to_multiclass(Y_test)
    Y_pred = to_multiclass(model.predict(X_test))
    Y_better_pred = to_multiclass(better_model.predict(X_better_test))

    Y_right = set(np.where(Y_true - Y_pred == 0)[0])
    Y_better_right = set(np.where(Y_true - Y_better_pred == 0)[0])

    return list(Y_better_right - Y_right)[:N_CORRECTED]
