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
import seaborn

from utility import *
from data_preprocessing import *


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
            tp_lst.append(np.sum(np.logical_and((iou_and_conf >= IOU_THRESH)[:, 0], (iou_and_conf >= thresh)[:, 1])))
            fp_lst.append(np.sum(np.logical_and((iou_and_conf < IOU_THRESH)[:, 0], (iou_and_conf >= thresh)[:, 1])))
            # tn_lst.append(np.sum(np.logical_and((iou_and_conf < IOU_THRESH)[:, 0], (iou_and_conf < thresh)[:, 1])))
            fn_lst.append(np.sum(np.logical_and((iou_and_conf >= IOU_THRESH)[:, 0], (iou_and_conf < thresh)[:, 1])))

        # Precision and recall for the PR curves.
        precision = [float(tp_lst[i]) / (tp_lst[i] + fp_lst[i]) if tp_lst[i] + fp_lst[i] != 0 else 0
                     for i in range(len(tp_lst))]
        recall = [float(tp_lst[i]) / (tp_lst[i] + fn_lst[i]) if tp_lst[i] + fn_lst[i] != 0 else 0
                  for i in range(len(tp_lst))]

        # Plot the precision against recall to get the PR curves.
        plt.plot(recall, precision)

        plt.title("PR Curves for Bird Detection with a IOU Threshold of " + str(IOU_THRESH))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(VISUALIZATION_DIR + "pr0%d.png" % (10 * IOU_THRESH))

        plt.show()

def visualize_f1_score(Y_true, Y_pred):
    """
    Visualize the F1 score for a given IOU threshold.
    """

    IOU_THRESH = IOU_THRESHES[0]

    # Array of F1 scores for each class.
    f1_scores = np.zeros((N_CLASSES, 2))
    f1_scores[:, 0] = np.arange(N_CLASSES)

    # Class information.
    classes_train, classes_test, classes_dict = load_classes()

    for species in range(N_CLASSES):
        # Find the subset of labels for a specific class.
        class_indices = np.where(classes_test == species)[0]
        Y_true_subset = Y_true[class_indices]
        Y_pred_subset = Y_pred[class_indices]

        # Array of IOU and confidence values.
        iou_and_conf = np.zeros((len(Y_true_subset), 2))

        iou_and_conf[:, 0] = [get_iou_of_boxes(Y_true_subset[i, :4], Y_pred_subset[i, :4]) for i in range(len(Y_true_subset))]
        iou_and_conf[:, 1] = Y_pred_subset[:, 4]

        # Lists for maximizing purposes.
        tp_lst = []
        fp_lst = []
        tn_lst = []
        fn_lst = []

        for thresh in np.arange(0, 1 + THRESH_INC, THRESH_INC):
            tp_lst.append(np.sum(np.logical_and((iou_and_conf >= IOU_THRESH)[:, 0], (iou_and_conf >= thresh)[:, 1])))
            fp_lst.append(np.sum(np.logical_and((iou_and_conf < IOU_THRESH)[:, 0], (iou_and_conf >= thresh)[:, 1])))
            # tn_lst.append(np.sum(np.logical_and((iou_and_conf < IOU_THRESH)[:, 0], (iou_and_conf < thresh)[:, 1])))
            fn_lst.append(np.sum(np.logical_and((iou_and_conf >= IOU_THRESH)[:, 0], (iou_and_conf < thresh)[:, 1])))

        # Precision and recall for the F1 score calculations.
        precision = [float(tp_lst[i]) / (tp_lst[i] + fp_lst[i]) if tp_lst[i] + fp_lst[i] != 0 else 0
                     for i in range(len(tp_lst))]

        recall = [float(tp_lst[i]) / (tp_lst[i] + fn_lst[i]) if tp_lst[i] + fn_lst[i] != 0 else 0
                  for i in range(len(tp_lst))]

        # Calculate F1 scores.
        # print(2 / (1 / np.array(precision) + 1 / np.array(recall)))
        f1_scores[species, 1] = max(2 / (1 / np.array(precision) + 1 / np.array(recall)))

    # Sort F1 scores.
    f1_scores = np.array(sorted(f1_scores, key=lambda x: x[1]))

    class_ids = np.arange(N_CLASSES)
    class_labels = [classes_dict[class_id] for class_id in f1_scores[:, 0]]
    frequencies = f1_scores[:, 1]

    plt.bar(class_ids, frequencies)

    plt.title("F1 Scores by Class for Bird Detection with a IOU Threshold of " + str(IOU_THRESH))
    # plt.xticks(class_ids, class_labels)
    plt.xticks([])
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.savefig(VISUALIZATION_DIR + "f1_bars_0%d.png" % (10 * IOU_THRESH))

    plt.show()