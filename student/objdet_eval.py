# -------------------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
# Copyright (C) 2022, Andreas Albrecht (modifications on starter code)
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# -------------------------------------------------------------------------------
#

# General package imports
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon
from operator import itemgetter

# Add project directory to python path to enable relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# Object detection tools and helper functions
import misc.objdet_tools as tools


## Compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
    """ Assess object detection performance by computing various performance measures.

    Args:
        - detections (list) : list of object detections
        - labels (list) : list of object labels
        - labals_valid (list) : list of valid object labels
        - min_iou (float) : minimum iou threshold, defaults to 0.5

    Returns:
        - det_performance (list) : detection performance measurement for the current frame
    """
    # Small constant added to the IoU denominator to avoid division by zero
    epsilon = 1.0e-15

    ## Find the best detection for each valid label using iou-based non-maximum suppresion
    ## in case of multiple possible associations of detections and labels
    true_positives = 0      # init no. of correctly detected (true positive (TP)) objects
    best_iou_matches = []   # init list of best_iou_matches
    center_devs = []        # init separate list of center deviations of best iou matches
    ious = []               # init separate list of iou values of best iou matches
    # loop over all labels
    for label_id, (label, valid) in enumerate(zip(labels, labels_valid)):

        # init list of possible detected object bounding box matches with the current label
        matches_lab_det = []

        # exclude all labels from statistics which are not considered valid
        if valid:

            ## Compute intersection over union (iou) and distance between centers

            ####### ID_S4_EX1 START #######
            #######
            print("student task ID_S4_EX1 ")

            ## Step 1 : Extract the four corners of the current label 2D bounding-box
            label_box_corners = tools.compute_box_corners(
                label.box.center_x, label.box.center_y, label.box.width, label.box.length,
                label.box.heading
            )

            ## Step 2 : Loop over all detected objects
            for det_id, det in enumerate(detections):

                ## Step 3 : Extract the four corners of the current detection 2D bounding-box
                _, det_box_center_x, det_box_center_y, det_box_center_z, \
                    _, det_box_width, det_box_length, det_box_heading = det
                det_box_corners = tools.compute_box_corners(
                    det_box_center_x, det_box_center_y, det_box_width, det_box_length,
                    det_box_heading
                )

                ## Step 4 : Compute x, y, z center distances between label & detection bounding-box
                dist_x = label.box.center_x - det_box_center_x
                dist_y = label.box.center_y - det_box_center_y
                dist_z = label.box.center_z - det_box_center_z

                ## Step 5 : Compute intersection over union (IOU) between label & detection
                ##          bounding-box
                label_box_polygon = Polygon(label_box_corners)
                det_box_polygon = Polygon(det_box_corners)
                intersection = label_box_polygon.intersection(det_box_polygon).area
                union = label_box_polygon.union(det_box_polygon).area
                iou = intersection / (union + epsilon)  # add epsilon in case of union == zero

                ## Step 6 : If IOU exceeds min_iou threshold, store [iou, dist_x, dist_y, dist_z]
                ##          as a potential label-detection bounding-box match in matches_lab_det
                if iou > min_iou:
                    matches_lab_det.append([iou, dist_x, dist_y, dist_z, label_id, det_id])

            #######
            ####### ID_S4_EX1 END #######

        ## Non-maximum suppresion (NMS): Find the best match for the current label and compute IOU
        ## and distance metrics for the best match, which is not yet associated as a best match with
        ## another label, and increae TP count
        if matches_lab_det:

            # sort potential matches by descending iou values (first column)
            matches_lab_det_sorted = sorted(matches_lab_det, key = lambda x : x[0])

            # loop over all potential matches starting with the highest iou value
            best_iou_match = []
            for match in matches_lab_det_sorted:
                # check if the detection id is already associated as a best match with another label
                if match[5] not in (_best_iou_match[5] for _best_iou_match in best_iou_matches):
                    # store current match as best iou match for the current label bounding-box
                    best_iou_match = match
                    # append best iou match to the list of best iou matches (true positive matches)
                    best_iou_matches.append(best_iou_match)
                    # append iou and center deviation of the best iou match to separate lists
                    ious.append(best_iou_match[0])              # append only iou
                    center_devs.append(best_iou_match[1:4])     # append only dist_x, dist_y, dist_z
                    # increase the TP count
                    true_positives += 1

    ####### ID_S4_EX2 START #######
    #######
    print("student task ID_S4_EX2")

    ## Step 1 : Compute the number of false positives (FP) or detections without a matching label
    false_positives = len(detections) - true_positives

    ## Step 2 : Compute the total number of positives (TP + FP) present in the scene
    all_positives = true_positives + false_positives

    ## Step 3 : Compute the number of false negatives (FN) or unmatched labels
    false_negatives = sum(labels_valid) - true_positives

    #######
    ####### ID_S4_EX2 END #######

    ## Prepare detection performance measurement results for evaluation
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]

    # Return detection performance
    return det_performance


## Evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all):

    # extract elements from detection performance measurement
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])

    ####### ID_S4_EX3 START #######
    #######
    print('student task ID_S4_EX3')

    ## Step 1 : Extract the total number of positives, true positives, false negatives,
    ##          and false positives from detection performance measurement
    pos_negs_arr = np.asarray(pos_negs)
    all_positives = np.sum(pos_negs_arr[:, 0])
    true_positives = np.sum(pos_negs_arr[:, 1])
    false_negatives = np.sum(pos_negs_arr[:, 2])
    false_positives = np.sum(pos_negs_arr[:, 3])
    print("TP+FP = " + str(all_positives) + "\n" + \
        ", TP    = " + str(true_positives) + "\n" + \
        ", FP    = " + str(false_positives) + "\n" + \
        ", FN    = " + str(false_negatives))

    ## Step 2 : Compute precision
    if all_positives == 0:
        print('There are no object detections. Set precision to zero.')
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)
        #precision = true_positives / all_positives

    ## Step 3 : Compute recall
    if (true_positives + false_negatives) == 0:
        print('There are no labels and no object detections. Set recall to zero.')
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    #######
    ####### ID_S4_EX3 END #######
    print('precision = ' + str(precision) + ", recall = " + str(recall))

    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for elem_tuple in center_devs:
        for elem in elem_tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)

    ## Compute statistics
    num_of_samples = len(ious_all)

    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__dev_x = np.std(devs_x_all)
    mean__dev_x = np.mean(devs_x_all)

    stdev__dev_y = np.std(devs_y_all)
    mean__dev_y = np.mean(devs_y_all)

    stdev__dev_z = np.std(devs_z_all)
    mean__dev_z = np.mean(devs_z_all)

    # Plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = [
        'detection precision',
        'detection recall',
        'intersection over union',
        'position errors in X',
        'position errors in Y',
        'position error in Z'
    ]
    textboxes = [
        '',
        '',
        '\n'.join((
            r'$\mathrm{mean}=%.4f$' % (mean__ious, ),
            r'$\mathrm{sigma}=%.4f$' % (stdev__ious, ),
            r'$\mathrm{n}=%.0f$' % (num_of_samples, )
        )),
        '\n'.join((
            r'$\mathrm{mean}=%.4f$' % (mean__dev_x, ),
            r'$\mathrm{sigma}=%.4f$' % (stdev__dev_x, ),
            r'$\mathrm{n}=%.0f$' % (num_of_samples, )
        )),
        '\n'.join((
            r'$\mathrm{mean}=%.4f$' % (mean__dev_y, ),
            r'$\mathrm{sigma}=%.4f$' % (stdev__dev_y, ),
            r'$\mathrm{n}=%.0f$' % (num_of_samples, )
            )),
        '\n'.join((
            r'$\mathrm{mean}=%.4f$' % (mean__dev_z, ),
            r'$\mathrm{sigma}=%.4f$' % (stdev__dev_z, ),
            r'$\mathrm{n}=%.0f$' % (num_of_samples, )
        ))
    ]
    _, axes = plt.subplots(2, 3)
    axes = axes.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, axis in enumerate(axes):
        axis.hist(data[idx], num_bins)
        axis.set_title(titles[idx])
        if textboxes[idx]:
            axis.text(
                0.05, 0.95, textboxes[idx], transform=axis.transAxes, fontsize=10,
                verticalalignment='top', bbox=props
            )
    plt.tight_layout()
    plt.show()
