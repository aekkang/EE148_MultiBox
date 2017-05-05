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
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects

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
EPOCHS = 32
VERBOSE = 1
SAVE = 1

# Visualization parameters
IOU_THRESHES = (0.5, 0.7, 0.9)
THRESH_INC = 0.005


##############################
# LOSS FUNCTIONS - 1x1
##############################

def F_loc(l, g):
    """
    The location loss function.
    """

    return tf.reduce_sum(tf.squared_difference(l, g)) / 2

def F_conf(c):
    """
    The confidence loss function.
    """

    return - tf.log(c)

def F(Y_true, Y_pred):
    """
    The total loss function.
    """
    return F_conf(Y_pred[:, :, :, 4]) + F_loc(Y_true[:, :, :, :4], Y_pred[:, :, :, :4])


##############################
# PRIOR FUNCTIONS - 1x1
##############################

def transform(Y):
    """
    Transforms the dataset with respect to priors. Data used to train the
    CNN should be transformed.
    """

    Y[:, :, :, 2:4] -= 1

def untransform(Y):
    """
    Undoes transform of the dataset with respect to priors. Data used for
    visualization should not be transformed.
    """

    Y[:, :, :, 2:4] += 1

def expand_box(box):
    """
    Transform the given bounding box to one with feasible coordinates.
    """

    new_box = [int(round(elem * INCEPTIONV3_SIZE)) for elem in box]

    return new_box


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
    
    # Overlay box.
    box = expand_box(box)
    if box[0] > box[2] or box[1] > box[3]:
        return

    # Add box.
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
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

def does_box_contain_point(box, point):
    """
    Determines whether a box contains a point.
    """

    p1x, p1y, p2x, p2y = box
    x, y = point

    return (p1x <= x <= p2x) and (p1y <= y <= p2y)

def do_boxes_intersect(box1, box2):
    """
    Determines whether two boxes intersect.
    """

    b1p1, b1p2 = box1[:2], box1[2:]
    b2p1, b2p2 = box2[:2], box2[2:]

    return does_box_contain_point(box2, b1p1) \
    or does_box_contain_point(box2, b1p2) \
    or does_box_contain_point(box2, (b1p1[0], b1p2[1])) \
    or does_box_contain_point(box2, (b1p2[0], b1p1[1])) \
    or does_box_contain_point(box1, b2p1) \
    or does_box_contain_point(box1, b2p2) \
    or does_box_contain_point(box1, (b2p1[0], b2p2[1])) \
    or does_box_contain_point(box1, (b2p2[0], b2p1[1]))

def get_area_of_box(box):
    """
    Determines the area of a box.
    """

    p1x, p1y, p2x, p2y = box

    return (p2x - p1x) * (p2y - p1y)

def get_intersection_of_boxes(box1, box2):
    """
    Determines the intersection area of two boxes.
    """

    if not do_boxes_intersect(box1, box2):
        return 0
    else:
        b1p1x, b1p1y, b1p2x, b1p2y = box1
        b2p1x, b2p1y, b2p2x, b2p2y = box2

        X = sorted([b1p1x, b1p2x, b2p1x, b2p2x])
        Y = sorted([b1p1y, b1p2y, b2p1y, b2p2y])

        return (X[2] - X[1]) * (Y[2] - Y[1])


def get_union_of_boxes(box1, box2):
    """
    Determines the union area of two boxes.
    """

    area1 = get_area_of_box(box1)
    area2 = get_area_of_box(box2)
    intersection = get_intersection_of_boxes(box1, box2)

    return area1 + area2 - intersection

def get_iou_of_boxes(box1, box2):
    """
    Determines the IOU of two boxes.
    """

    intersection = get_intersection_of_boxes(box1, box2)
    union = get_union_of_boxes(box1, box2)
    return float(intersection) / union


##############################
# MISCELLANEOUS FUNCTIONS
##############################

def load_custom_model(filename):
    """
    Load a model with a custom loss function.
    """

    get_custom_objects().update({"F": F})
    model = load_model(RESULTS_DIR + filename)

    return model
