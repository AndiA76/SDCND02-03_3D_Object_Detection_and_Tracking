# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : helper functions for loop_over_dataset.py
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
# Starter code modified by Andreas Albrecht, 2022
# ----------------------------------------------------------------------
#

# imports
import os
import pickle

## Saves an object to a binary file
def save_object_to_file(object, file_path, base_filename, object_name, frame_id=1):
    """ Save object to a binary file.
    Args:
        - object (pythonObject): object (e.g. a lidar point cloud)
        - file_path (str) : filepath to binary file
        - base_filename (str): base filename
        - object_name (str): object name
        - frame_id (int): frame id
    Returns:
        - (list): object (e.g. lidar point cloud)
    """
    object_filename = os.path.join(file_path, os.path.splitext(base_filename)[0]
                                   + "__frame-" + str(frame_id) + "__" + object_name + ".pkl")
    with open(object_filename, 'wb') as f:
        pickle.dump(object, f)

## Loads an object from a binary file
def load_object_from_file(file_path, base_filename, object_name, frame_id=1):
    """ Load object from binary file
    Args:
        - file_path (str) : filepath to binary file
        - base_filename (str): base filename
        - object_name (str): object name
        - frame_id (int): frame id
    Returns:
        - (list): object (e.g. lidar point cloud)
    """
    object_filename = os.path.join(file_path, os.path.splitext(base_filename)[0]
                                   + "__frame-" + str(frame_id) + "__" + object_name + ".pkl")
    with open(object_filename, 'rb') as f:
        object = pickle.load(f)
        return object

## Prepares an exec_list with all tasks to be executed
def make_exec_list(exec_data, exec_detection, exec_tracking, exec_visualization):
    """ Make an overall execution list from all task lists provided as input arguments.
    Args:
        - exec_data (list) : data processing task list
        - exec_detection (list) : object detection task list
        - exec_tracking (list) : object tracking task list
        - exec_visualization (list) : visualization task list
    Returns:
        - exec_list (list) : list with all execuation tasks
    """
    # save all tasks in exec_list
    exec_list = exec_data + exec_detection + exec_tracking + exec_visualization

    # check if we need pcl
    if any(i in exec_list for i in ('validate_object_labels', 'bev_from_pcl')):
        exec_list.append('pcl_from_rangeimage')
    # check if we need image
    if any(
        i in exec_list for i in (
        'show_tracks', 'show_labels_in_image', 'show_objects_in_bev_labels_in_camera'
        )
    ):
        exec_list.append('load_image')
    # movie does not work without show_tracks
    if 'make_tracking_movie' in exec_list:
        exec_list.append('show_tracks')
    # return overall task list
    return exec_list