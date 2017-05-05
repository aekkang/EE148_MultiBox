##################################################
# EE 148 Assignment 4
#
# Author:   Andrew Kang
# File:     multibox.py
# Desc:     Implements a MultiBox detector
#           to predict on the Caltech-UCSD
#           Birds-200 dataset.
##################################################

import numpy as np
import matplotlib.pyplot as plt
import cv2

from utility import *


##############################
# Loading
##############################

# Load predictions.
red_boxes_by_id, green_boxes_by_id = load_boxes(RESULTS_PATH, False, key="light_pred")
red_boxes = [[] for _ in range(TEST_SIZE)]
green_boxes = [[] for _ in range(TEST_SIZE)]

# Accumulate predictions.
for n in range(TEST_SIZE):
    for red_box in red_boxes_by_id:
        if red_box[0] == n:
            red_box_int = np.uint16(np.around(red_box[1:5]))
            score = red_box[5]
            red_boxes[n].append((red_box_int, score))

    for green_box in green_boxes_by_id:
        if green_box[0] == n:
            green_box_int = np.uint16(np.around(green_box[1:5]))
            score = green_box[5]
            green_boxes[n].append((green_box_int, score))

# Load annotations.
red_anns = []
green_anns = []

# Accumulate annotations.
for n in range(TEST_SIZE):
    filename = find_filename(n)
    red_anns_by_file, green_anns_by_file = load_boxes(ANN_DIR + filename + ".mat")
    red_anns.append(red_anns_by_file)
    green_anns.append(green_anns_by_file)

##############################
# PR CURVES
##############################

def visualize_pr_curves(Y_true, Y_pred):
    """
    Visualize the PR curves for different IOU thresholds.
    """

    # Array of IOU and confidence values.
    iou_and_conf = np.zeros((len(Y_true), 2))

    iou_and_conf[:, 0] = [get_iou_of_boxes(Y_true[i, :4], Y_pred[i, :4]) for i in range(len(Y_true))]
    iou_and_conf[:, 1] = Y_pred[:, 4]

    for IOU_THRESH in IOU_THRESHES:
        # Lists for graphing purposes.
        tp_lst = []
        fp_lst = []
        tn_lst = []
        fn_lst = []

        for thresh in np.arange(0, 1 + THRESH_INC, THRESH_INC):
            tp_lst.append(np.logical_and((iou_and_conf > IOU_THRESH)[0], (iou_and_conf > thresh)[1]))
            fp_lst.append(np.logical_and((iou_and_conf < IOU_THRESH)[0], (iou_and_conf > thresh)[1]))
            tn_lst.append(np.logical_and((iou_and_conf < IOU_THRESH)[0], (iou_and_conf < thresh)[1]))
            fn_lst.append(np.logical_and((iou_and_conf > IOU_THRESH)[0], (iou_and_conf < thresh)[1]))

        # Precision and recall for the PR curves.
        precision = [float(tp_lst[i]) / (tp_lst[i] + fp_lst[i]) if tp_lst[i] + fp_lst[i] != 0 else 0
                     for i in range(len(tp_lst))]
        recall = [float(tp_lst[i]) / (tp_lst[i] + fn_lst[i]) if tp_lst[i] + fn_lst[i] != 0 else 0
                  for i in range(len(tp_lst))]

        # Plot the precision against recall to get the PR curves.
        plt.plot(recall, precision)

        plt.title("PR Curves for Traffic Light Detection with a IOU Threshold of " + str(IOU_THRESH))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        # plt.savefig(VISUALIZATION_DIR + "pr0%d.png" % (10 * IOU_THRESH))

        plt.show()
