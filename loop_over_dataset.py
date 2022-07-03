# -------------------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
# Copyright (C) 2022, Andreas Albrecht (modifications on starter code)
#
# Purpose of this file : Loop over all frames in a Waymo Open Dataset file,
#                        detect and track objects and visualize results
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# -------------------------------------------------------------------------------
#

##################
## Imports

## General package imports
import os
import sys
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import copy
import torch

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2

## 3d object detection
import student.objdet_pcl as pcl
import student.objdet_detect as det
import student.objdet_eval as eval

import misc.objdet_tools as tools
from misc.helpers import save_object_to_file, load_object_from_file, make_exec_list

## Tracking
from student.filter import Filter
from student.trackmanagement import Trackmanagement
from student.association import Association
from student.measurements import Sensor, Measurement
from misc.evaluation import plot_tracks, plot_rmse, make_gif_movie, make_movie
import misc.params as params

##################
## Set parameters and perform initializations

## Select Waymo Open Dataset file and frame numbers
# Select data sequence to evaluate on
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
#data_filename = 'training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord' # Sequence 2
#data_filename = 'training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord' # Sequence 3
# Show only frames in selected interval for debugging
show_only_frames = [0, 198]  # e.g. [0, 1], [65, 100], [0, 198], [0, 200], ...

## Prepare Waymo Open Dataset file for loading
# Use an adjustable path in case this script is called from another working directory
data_fullpath = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename
)
results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
# Initialize Waymo data file reader
datafile = WaymoDataFileReader(data_fullpath)
# Initialize dataset iterator
datafile_iter = iter(datafile)

## Initialize object detection
# Initialize object detection model
configs_det = det.load_configs(model_name='darknet') # options: 'darknet', 'fpn_resnet'
model_det = det.create_model(configs_det)
# Set True to use groundtruth labels as objects, or False to use model-based detections
configs_det.use_labels_as_objects = False
# Save results to file (based on data_filename)
configs_det.save_results = False

## Uncomment this setting to restrict the y-range in the final project
configs_det.lim_y = [-25, 25] # [-25, 25]

## Initialize tracking
# Set up Kalman filter
KF = Filter()
# Init data association
association = Association(association_method='GNN')  # association_method = 'SNN' or 'GNN'
# Init track manager
manager = Trackmanagement()
# Init lidar sensor object
lidar = None
# Init camera sensor object
camera = None
# Make random values predictable by setting a fixed seed
np.random.seed(10)

## Selective execution and visualization
# Set exec_data options:
#   'load_image', 'pcl_from_rangeimage'
exec_data = [
    'load_image',
    'pcl_from_rangeimage',
]
# exec_detection options (options not in list will be loaded from file):
#   'bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'
exec_detection = [
    #'bev_from_pcl',
    #'detect_objects',
    #'validate_object_labels',
    #'measure_detection_performance',
]
# Set exec_tracking options:
#   'perform_tracking'
exec_tracking = [
    'perform_tracking',
]
# Set exec_visualization options:
#   'show_range_image', 'show_bev', 'show_pcl', 'show_bev_from_pcl', 'show_labels_in_image',
#   'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera',
#   'show_detection_performance', 'show_tracks', 'make_tracking_movie'
exec_visualization = [
    #'show_range_image',
    #'show_bev',
    #'show_pcl',
    #'show_bev_from_pcl',
    #'show_labels_in_image',
    #'show_objects_and_labels_in_bev',
    #'show_objects_in_bev_labels_in_camera',
    #'show_detection_performance',
    'show_tracks',
    'make_tracking_movie',
]
# Join all options to an execution setting list
exec_list = make_exec_list(exec_data, exec_detection, exec_tracking, exec_visualization)

# Set pause time between frames in ms (0 = stop between frames until key is pressed)
#vis_pause_time = 0
vis_pause_time = 2000

##################
## Perform object detection & tracking over all selected frames
cnt_frame = 0
all_labels = []
det_performance_all = []
# Make random values predictable by setting a fixed seed
np.random.seed(0)
# Init track plot if 'show_tracks' option is set
if 'show_tracks' in exec_list:
    fig, (ax2, ax) = plt.subplots(1,2)

# Start loop over the dataset
while True:
    try:
        ## Get next frame from Waymo dataset
        frame = next(datafile_iter)
        if cnt_frame < show_only_frames[0]:
            cnt_frame += 1
            continue
        elif cnt_frame > show_only_frames[1]:
            print('reached end of selected frames')
            break

        print('------------------------------')
        print('processing frame #' + str(cnt_frame))

        #################################
        ## Perform 3D object detection

        ## Extract lidar and camera calibration data from the data frame
        # Extract lidar calibration data for the roof top lidar sensor
        lidar_name = dataset_pb2.LaserName.TOP
        lidar_calibration = waymo_utils.get(frame.context.laser_calibrations, lidar_name)
        # Extract camera calibration data for the front camera
        camera_name = dataset_pb2.CameraName.FRONT
        camera_calibration = waymo_utils.get(frame.context.camera_calibrations, camera_name)

        ## Extract the front camera image from the current data frame
        if 'load_image' in exec_list:
            image = tools.extract_front_camera_image(frame)

        ## Compute lidar point-cloud from range image
        if 'pcl_from_rangeimage' in exec_list:
            print('computing point-cloud from lidar range image')
            lidar_pcl = tools.pcl_from_range_image(frame, lidar_name)
            # save lidar point-cloud to file (based on data_filename)
            if configs_det.save_results:
                save_object_to_file(
                    lidar_pcl, results_fullpath, data_filename, 'lidar_pcl', cnt_frame
                )
        else:
            print('loading lidar point-cloud from result file')
            lidar_pcl = load_object_from_file(
                results_fullpath, data_filename, 'lidar_pcl', cnt_frame
            )

        ## Compute lidar birds-eye view (bev)
        if 'bev_from_pcl' in exec_list:
            print('computing birds-eye view from lidar point cloud')
            lidar_bev = pcl.bev_from_pcl(
                lidar_pcl,
                configs_det,
                visualize=('show_bev_from_pcl' in exec_visualization),
                pause_time_ms=vis_pause_time
            )
            # save bev map obtained from lidar point-cloud to file (based on data_filename)
            if configs_det.save_results:
                save_object_to_file(
                    lidar_bev, results_fullpath, data_filename, 'lidar_bev', cnt_frame
                )
        else:
            print('loading birds-eve view from result file')
            lidar_bev = load_object_from_file(
                results_fullpath, data_filename, 'lidar_bev', cnt_frame
            )
            lidar_bev = torch.FloatTensor(lidar_bev)

        ## 3D object detection
        if configs_det.use_labels_as_objects is True:
            print('using groundtruth labels as objects')
            detections = tools.convert_labels_into_objects(frame.laser_labels, configs_det)
        else:
            if 'detect_objects' in exec_list:
                print('detecting objects in lidar pointcloud')
                detections = det.detect_objects(lidar_bev, model_det, configs_det)
                # save object detections from lidar bev map to file (based on data_filename)
                if configs_det.save_results:
                    save_object_to_file(
                        detections, results_fullpath, data_filename, 'detections', cnt_frame
                    )
            else:
                print('loading detected objects from result file')
                # load different data for final project vs. mid-term project
                if 'perform_tracking' in exec_list:
                    detections = load_object_from_file(
                        results_fullpath, data_filename, 'detections', cnt_frame
                    )
                else:
                    detections = load_object_from_file(
                        results_fullpath, data_filename,
                        'detections_' + configs_det.arch + '_' + str(configs_det.conf_thresh),
                        cnt_frame
                    )

        ## Validate object labels
        if 'validate_object_labels' in exec_list:
            print("validating object labels")
            valid_label_flags = tools.validate_object_labels(
                frame.laser_labels, lidar_pcl, configs_det,
                0 if configs_det.use_labels_as_objects is True else 10
            )
            # save valid object label flags to file (based on data_filename)
            if configs_det.save_results:
                save_object_to_file(
                    valid_label_flags, results_fullpath, data_filename, 'label_labels', cnt_frame
                )
        else:
            print('loading object labels and validation from result file')
            valid_label_flags = load_object_from_file(
                results_fullpath, data_filename, 'valid_labels', cnt_frame
            )

        ## Performance evaluation for object detection
        if 'measure_detection_performance' in exec_list:
            print('measuring detection performance')
            det_performance = eval.measure_detection_performance(
                detections, frame.laser_labels, valid_label_flags, configs_det.iou_thresh
            )
            # save object detection performance to file (based on data_filename)
            if configs_det.save_results:
                save_object_to_file(
                    det_performance, results_fullpath, data_filename,
                    'det_performance_' + configs_det.arch + '_' + str(configs_det.conf_thresh),
                    cnt_frame
                )
        else:
            print('loading detection performance measures from file')
            # load different data for final project vs. mid-term project
            if 'perform_tracking' in exec_list:
                det_performance = load_object_from_file(
                    results_fullpath, data_filename, 'det_performance', cnt_frame
                )
            else:
                det_performance = load_object_from_file(
                    results_fullpath, data_filename,
                    'det_performance_' + configs_det.arch + '_' + str(configs_det.conf_thresh),
                    cnt_frame
                )

        ## store all evaluation results in a list for performance assessment at the end
        det_performance_all.append(det_performance)


        ## Visualization for object
        # show 8-bit range image including range channel (top) and intensity channel (bottom)
        if 'show_range_image' in exec_list:
            img_range_8bit = pcl.show_range_image(frame, lidar_name, configs_det)
            cv2.imshow('range_image', img_range_8bit)
            cv2.waitKey(vis_pause_time)

        # show original 3D point cloud (only if 'show_bev_from_pcl' is not active)
        if 'show_pcl' in exec_list and not 'show_bev_from_pcl' in exec_list:
            # Alternative 1:
            if "o3d_pcl_vis" not in globals():
                o3d_pcl_vis = pcl.PointCloudVisualizer(
                    lidar_pcl, window_name='Show 3D Lidar Point-Cloud', pause_time_ms=vis_pause_time)
            else:
                o3d_pcl_vis.update(lidar_pcl)
            # Alternative 2:
            #pcl.show_pcl(
            #    lidar_pcl,
            #    window_name='Show 3D Lidar Point Cloud',
            #    pause_time_ms=vis_pause_time
            #)

        # show point cloud in 2D bird's eye view
        if 'show_bev' in exec_list:
            tools.show_bev(lidar_bev, configs_det)
            cv2.waitKey(vis_pause_time)

        # show image labels projected onto the camera image
        if 'show_labels_in_image' in exec_list:
            img_labels = tools.project_labels_into_camera(
                camera_calibration, image, frame.laser_labels, valid_label_flags, 0.5
            )
            cv2.imshow('img_labels', img_labels)
            cv2.waitKey(vis_pause_time)

        # show objects and object labels in point cloud bird's eye view
        if 'show_objects_and_labels_in_bev' in exec_list:
            tools.show_objects_labels_in_bev(
                detections, frame.laser_labels, lidar_bev, configs_det
            )
            cv2.waitKey(vis_pause_time)

        # show objects detected from birds-eye-view on 3D lidar point cloud in the camera image
        if 'show_objects_in_bev_labels_in_camera' in exec_list:
            tools.show_objects_in_bev_labels_in_camera(
                detections, lidar_bev, image, frame.laser_labels, valid_label_flags,
                camera_calibration, configs_det
            )
            cv2.waitKey(vis_pause_time)


        #################################
        ## Perform tracking
        if 'perform_tracking' in exec_list:
            # set up sensor objects
            if lidar is None:
                lidar = Sensor('lidar', lidar_calibration)
            if camera is None:
                camera = Sensor('camera', camera_calibration)

            # preprocess lidar detections
            meas_list_lidar = []
            for detection in detections:
                # check if measurement lies inside specified range
                if detection[1] > configs_det.lim_x[0] and detection[1] < configs_det.lim_x[1] and \
                    detection[2] > configs_det.lim_y[0] and detection[2] < configs_det.lim_y[1]:
                    meas_list_lidar = lidar.generate_measurement(
                        cnt_frame, detection[1:], meas_list_lidar
                    )

            # preprocess camera detections
            meas_list_cam = []
            for label in frame.camera_labels[0].labels:
                if label.type == label_pb2.Label.Type.TYPE_VEHICLE:

                    box = label.box
                    # use camera labels as measurements and add some random noise
                    z = [box.center_x, box.center_y, box.width, box.length]
                    z[0] = z[0] + np.random.normal(0, params.sigma_cam_i)
                    z[1] = z[1] + np.random.normal(0, params.sigma_cam_j)
                    meas_list_cam = camera.generate_measurement(cnt_frame, z, meas_list_cam)

            # Kalman prediction
            for track in manager.track_list:
                print('predict track', track.id)
                KF.predict(track)
                track.set_t((cnt_frame - 1)*0.1) # save next timestamp

            # associate all lidar measurements to all tracks
            association.associate_and_update(manager, meas_list_lidar, KF)

            # associate all camera measurements to all tracks
            association.associate_and_update(manager, meas_list_cam, KF)

            # save results for evaluation
            result_dict = {}
            for track in manager.track_list:
                result_dict[track.id] = track
            manager.result_list.append(copy.deepcopy(result_dict))
            label_list = [frame.laser_labels, valid_label_flags]
            all_labels.append(label_list)

            # visualization
            if 'show_tracks' in exec_list:
                fig, ax, ax2 = plot_tracks(
                    fig, ax, ax2, manager.track_list, meas_list_lidar, frame.laser_labels,
                    valid_label_flags, image, camera, configs_det
                )
                if 'make_tracking_movie' in exec_list:
                    # save track plots to file
                    fname = f'{results_fullpath}/tracking{cnt_frame:03d}.png'
                    #fname = results_fullpath + '/tracking%03d.png' % cnt_frame
                    print('Saving frame', fname)
                    fig.savefig(fname)

        # increment frame counter
        cnt_frame += 1

    except StopIteration:
        # if StopIteration is raised, break from loop
        print("StopIteration has been raised\n")
        break

# Clean up visualizer object if it exists
if "o3d_pcl_vis" in globals():
    del o3d_pcl_vis

#################################
## Post-processing

## Evaluate object detection performance
if 'show_detection_performance' in exec_list:
    eval.compute_performance_stats(det_performance_all)

## Plot RMSE for all tracks
if 'show_tracks' in exec_list:
    plot_rmse(manager, all_labels, configs_det)

## Make movie from tracking results
if 'make_tracking_movie' in exec_list:
    # make gif movie
    make_gif_movie(results_fullpath, clean_up=False)
    # make mp4 or avi movie
    make_movie(results_fullpath, video_format='mp4', clean_up=False)
