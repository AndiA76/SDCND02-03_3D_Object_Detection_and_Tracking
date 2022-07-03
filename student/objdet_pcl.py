# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
# Copyright (C) 2022, Andreas Albrecht (modifications on starter code)
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import os
import sys
import zlib
import cv2
import numpy as np
import open3d as o3d
import time
import torch

# add project directory to python path to enable relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
#from functools import partial
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools


## Visualize lidar point-cloud (variant 1: class-based implementation)
class PointCloudVisualizer():
    """ Visualizer class to show lidar point-clouds frame by frame in 3D. """
    def __init__(self, pcl, window_name='Show 3D Lidar Point-Cloud',
        width=1280, height=720, left=50, top=50, pause_time_ms=0):
        """ Class constructor: Create a visualizer object showing lidar point-clouds frame
            by frame in 3D. If pause time is zero the window is closed and the next lidar
            frame is shown when the right arrow key is pressed. Otherwise the window is
            updated automatically with the next lidar frame (keeping window settings and
            view perspective) after a delay defined by pause time (plus processing time).

        Args:
            - pcl : 3D lidar point cloud
            - window_name (str) : Window name. Defaults to 'Show 3D Lidar Point-Cloud'.
            - width (int) : Width of the visualizer window. Default to 1280.
            - height (int) : Height of the visualizer window. Default to 720.
            - left (int) : horizontal cooridnate of the upper left visualizer window corner.
              Defaults to 50.
            - upper (int) : Vertical coordinate of the upper left visualizer window corner.
              Defaults to 50.
            - pause_time_ms (float) : Time to pause before an update of the visualization.
              Defaults to 0.
        """
        ####### ID_S1_EX2 START #######
        #######
        print("student task ID_S1_EX2 (show_pcl)")

        # http://www.open3d.org/docs/0.8.0/tutorial/Advanced/non_blocking_visualization.html

        ## Step 1 : Initialize open3d with key callback and create window
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        # Set window name
        self.window_name = window_name
        # Set window size defined by upper left corner, width and height
        self.width = width
        self.height = height
        self.top = top
        self.left = left
        # Set pause time in seconds
        self.pause_time_s = pause_time_ms/1000
        # Create a visualization window and set its properties
        self.vis.create_window(
            window_name=self.window_name,
            width=width, height=self.height, left=self.left, top=self.top, visible=True
        )
        ## Define renderer settings
        # self.vis.get_render_option().background_color = np.array([0., 0., 0.]) # black
        self.vis.get_render_option().background_color = np.array([1., 1., 1.]) # white
        self.vis.get_render_option().point_size = 2
        self.vis.get_render_option().mesh_show_back_face = True

        ## Step 2 : Create instance of open3d point-cloud class
        self.pcd = o3d.geometry.PointCloud()

        ## Step 3 : Set points in pcd instance by converting the point-cloud into 3d vectors
        ##          taking the 3d point coordinates, dropping intensity
        self.pcd.points = o3d.utility.Vector3dVector(pcl[:,:3])

        ## Step 4 : For the first frame, add the 3d pcd instance to visualization using
        ##          add_geometry; for all other frames, use update_geometry instead
        self.vis.add_geometry(self.pcd)

        ## Step 5 : Visualize point cloud and keep window open until right-arrow is pressed
        ##          (key-code for right arrow: 262)
        # Callback function for closing the visualizer window on right arrow key press
        def close_on_key_callback(vis_obj):
            """ Close window routine """
            vis_obj.close()
            return True
        # Register key callback function "close on right-arrow key press"
        self.vis.register_key_callback(262, close_on_key_callback)
        print('adjust view with the mouse and press right-arrow key to close figure and continue ...')
        # activate the visualizer window to wait for key callback
        self.vis.run()  # key callback seems to work only in this case, but not for the updates any more

    def __del__(self):
        """ Class destructor: Destroy visualizer window. """
        # Destroy visualizer window
        self.vis.destroy_window()

    def update(self, pcl):
        """" Update point cloud visualization waiting for key callback or pausing for a moment. """
        # Update points in pcd instance by converting the point-cloud into 3d vectors
        # taking the 3d point coordinates, dropping intensity
        self.pcd.points = o3d.utility.Vector3dVector(pcl[:,:3])
        # Update geometry data to create a new visualization
        self.vis.update_geometry(self.pcd)
         # Call the renderer to update the visualization (not waiting for key callback)
        self.vis.update_renderer()
        # Poll for events
        self.vis.poll_events()
        if self.pause_time_s <= 0:
            # Visualize point cloud and keep window open until right-arrow is pressed
            print('press right-arrow key to close figure and continue ...')
            # Activate the visualizer window waiting for key callback
            self.vis.run()  # key callback seems not to work => graph is updated without waiting for callback
        else:
            # Pause before updating the visiualizer window with the next frame
            time.sleep(self.pause_time_s)
    #######
    ####### ID_S1_EX2 END #######


## Visualize lidar point-cloud (variante 2: function-based implementation)
def show_pcl(pcl, window_name='Show 3D Lidar Point-Cloud', pause_time_ms=0):
    """ Visualize lidar point-cloud frame by frame in 3D. If pause time is zero the
        window is closed and the next lidar frame is shown when the right arrow key is
        pressed. Otherwise the window is updated automatically with the next lidar
        frame (keeping window settings and view perspective) after a delay given by
        pause time plus processing time.

    Args:
        - pcl : 3D lidar point cloud
        - window_name (str) : window name. Defaults to 'Show 3D Lidar Point-Cloud'.
        - pause_time_ms (float) : time to pause before an update of the visualization.
          Defaults to 0.
    """

    ####### ID_S1_EX2 START #######
    #######
    print("student task ID_S1_EX2 (show_pcl)")

    # http://www.open3d.org/docs/0.8.0/tutorial/Advanced/non_blocking_visualization.html

    # Remark: When adding the visualizer as an attribute to the function all updates will
    # be plotted to the same window. So this solution should not be used if independent
    # visualizer instances are needed, e.g. to display the original 3D point cloud and a
    # 2D projection of the 3D point cloud in two separate windows in parallel. This should
    # be done in an either-or manner. 

    # attribute show_pcl() function with an o3d visualizer object if it does not already exist
    if not hasattr(show_pcl, 'vis'):
        ## Step 1 : Initialize open3d with key callback and create window
        show_pcl.vis = o3d.visualization.VisualizerWithKeyCallback()
        # create a visualization window and set its properties
        show_pcl.vis.create_window(
            window_name=window_name,
            width=1280, height=720, left=50, top=50, visible=True
        )
        ## renderer settings
        #vis.get_render_option().background_color = np.array([0., 0., 0.]) # black
        show_pcl.vis.get_render_option().background_color = np.array([1., 1., 1.]) # white
        show_pcl.vis.get_render_option().point_size = 2
        show_pcl.vis.get_render_option().mesh_show_back_face = True

        ## Step 2 : Create instance of open3d point-cloud class
        show_pcl.pcd = o3d.geometry.PointCloud()

        ## Step 3 : Set points in pcd instance by converting the point-cloud into 3d vectors
        ##          (use open3d function Vector3dVector) take point coordinates, drop intensity
        show_pcl.pcd.points = o3d.utility.Vector3dVector(pcl[:,:3])

        ## Step 4 : For the first frame, add the pcd instance to visualization using add_
        ##          geometry; for all other frames, use update_geometry instead
        show_pcl.vis.add_geometry(show_pcl.pcd)

        ## Step 5 : Visualize point cloud and keep window open until right-arrow is pressed
        ##          (key-code for right arrow: 262)
        def close_on_key_callback(vis_obj):
            """ Close window key callback function. """
            vis_obj.close()
            return True
        # close on right-arrow key press
        show_pcl.vis.register_key_callback(262, close_on_key_callback)
        print('adjust the view with the mouse and press right-arrow key to close figure and continue ...')
        # activate the visualizer window waiting for key callback
        show_pcl.vis.run()
    else:
        ## Step 6: Call the renderer to update the visualization if show_pcl function is
        ##         attributed with an o3d visualizer object

        # Update points in pcd instance by converting the point-cloud into 3d vectors
        # taking the 3d point coordinates, dropping intensity
        show_pcl.pcd.points = o3d.utility.Vector3dVector(pcl[:,:3])
        # Update geometry data to create a new visualization
        show_pcl.vis.update_geometry(show_pcl.pcd)

        if pause_time_ms <= 0:
            # Visualize point cloud and keep window open until right-arrow is pressed
            print('press right-arrow key to close figure and continue ...')
            # Call the renderer to update the visualization (not waiting for key callback)
            show_pcl.vis.update_renderer()
            # Poll for events
            show_pcl.vis.poll_events()
            # Activate the visualizer window waiting for key callback
            show_pcl.vis.run()
        else:
            # Call the renderer to update the visualization (not waiting for key callback)
            show_pcl.vis.update_renderer()
            # Poll for events
            show_pcl.vis.poll_events()
            # Pause a user-defined period of time in seconds
            pause_time_s = pause_time_ms/1000
            time.sleep(pause_time_s)
    #######
    ####### ID_S1_EX2 END #######


## Load range image from lidar data frame
def load_range_image(frame, lidar_name):
    """ Extract range image from the lidar data frame (first response) of the roof-mounted lidar.
        The first return response of the emitted lidar signal has 6 channels, namely, range,
        intensity, elongation, x, y, and z. The x, y, and z lidar point coordinates are measured
        in vehicle frame. Pixels with range 0 are not valid points.

    Args:
        - frame (lidar frame) : lidar data frame
        - lidar_name (str) : name of the lidar sensor

    Returns:
        - range_image (dataset_pb2.MatrixFloat()) : first return lidar response range image with
            all 6 channels
    """

    # get lidar data structure from frame
    #(range_image, camera_projection, range_image_top_poses) = \
    #    waymo_utils.parse_range_image_and_camera_projection(frame, second_response=False)
    lidar_frame = [obj for obj in frame.lasers if obj.name == lidar_name][0]

    # use first lidar response
    range_image = []
    if len(lidar_frame.ri_return1.range_image_compressed) > 0:
        range_image = dataset_pb2.MatrixFloat()
        range_image.ParseFromString(
            zlib.decompress(lidar_frame.ri_return1.range_image_compressed)
        )
        range_image = np.array(range_image.data).reshape(range_image.shape.dims)

    # return first return lidar response range image with all 6 channels
    return range_image


## Center crop a symmetric field of view section relative to the driving direction
def center_crop_fov_section(stacked_range_image, fov_angle=360):
    """ Crop a symmetric field of view section from the stacked 360° surround view
        range iamge with +/-fov_angle/2 left and right of the forward facing y-axis.

    Args:
        - stacked_range_iamge (np.ndarray) : 2D stacked range image
        - fov_angle (int) : field-of-view-angle of the crop section, defaults to 360
            (360 => meaning no cropping)

    Returns:
        - center_crop (np.ndarray) : fov crop of the stacked range image
    """

    if 0 < fov_angle < 360:
        # get index of the fov center which maps to the forward facing y-axis of the car
        fov_center_idx = int(stacked_range_image.shape[1] / 2)
        # get index increment of a half fov angle crop section
        fov_halfangle_idx_incr = int(stacked_range_image.shape[1] * fov_angle/360/2)
        # get left and right index of the crop section
        fov_left_idx = fov_center_idx - fov_halfangle_idx_incr
        fov_right_idx = fov_center_idx + fov_halfangle_idx_incr
        # take center fov crop from the stacked range image
        center_crop = stacked_range_image[:, fov_left_idx:fov_right_idx]
    elif fov_angle == 360:
        # no fov crop is taken from stacked range image
        center_crop = stacked_range_image
    else:
        print('Error in center_crop_fov_section(): fov_angle out of bounds ]0...360]!')
        # no fov crop is taken from stacked range image
        center_crop = stacked_range_image

    # return center fov crop of the stacked range image
    return center_crop


## Visualize range image
def show_range_image(frame, lidar_name, configs):
    """ Show stacked range and intensity image of the first return lidar response converted to
        8bit integers. The first return response of the emitted lidar signal has six channels,
        namely, range, intensity, elongation, x, y, and z. The x, y, and z lidar point coordinates
        are measured in vehicle frame. Pixels with range 0 are not valid points. Elongation is the
        change in range measurement due to the relative motion of the car superposed to the lidar
        measurement.

    Args:
        - frame (lidar frame) : lidar data frame
        - lidar_name (str) : name of the lidar sensor
        - configs (easydict.EasyDict) : configuration data

    Returns:
        - img_range_intensity (8-bit unit nd.array) : stacked 8-bit range and intensity image
    """

    ####### ID_S1_EX1 START #######
    #######
    print("student task ID_S1_EX1 (show_range_image)")

    ## Step 1 : Extract lidar data and 6-channel range image for the roof-mounted lidar
    range_image = load_range_image(frame, lidar_name)

    ## Step 2 : Extract the range and the intensity channel from the range image ri
    ##          extract range channel
    img_range = range_image[:,:,0]
    # extract intensity channel
    img_intensity = range_image[:,:,1]
    # extract elongation channel (elongation due to Doppler effect)
    #img_elongation = range_image[:,:,2]

    ## Step 3 : Set all invalid values < 0 to zero (points with range zero are invalid points)
    img_range[np.where(img_range < 0)] = 0.0
    img_intensity[np.where(img_intensity < 0)] = 0.0

    ## Step 4 : Map the range channel onto an 8-bit scale and make sure that the full range of
    ##          values is appropriately considered
    # normalize range channel of the range image and map to 8-bit scale
    img_range = 255 * img_range / (np.amax(img_range) - np.amin(img_range))

    ## Step 5 : Map the intensity channel onto an 8-bit scale and normalize with the difference
    ##          between the 1- and 99-percentile to mitigate the influence of outliers
    lower_percentile = np.percentile(img_intensity, configs.intensity_lower_percentile)
    upper_percentile = np.percentile(img_intensity, configs.intensity_upper_percentile)
    img_intensity = 255 * (
        np.clip(img_intensity, lower_percentile, upper_percentile) - lower_percentile
    ) / (upper_percentile - lower_percentile)

    # alternatively: normalize intensity channel of the range image with 50% of the relative
    # maximum intensity and map to 8-bit scale
    #img_intensity = 255 * img_intensity * np.amax(img_intensity)/2 / \
    #    (np.amax(img_intensity) - np.amin(img_intensity))

    # Step 6 : Stack the range and intensity image vertically using np.vstack and convert the
    #          result to an unsigned 8-bit integer
    img_range_intensity = np.vstack((img_range, img_intensity)).astype(np.uint8)

    ## Crop range image to +/- 90 degree left and right of the forward-facing y-axis
    # set field of view (fov) section to be extracted from 360° range image
    img_range_intensity = center_crop_fov_section(img_range_intensity, fov_angle=180)
    #######
    ####### ID_S1_EX1 END #######

    # return stacked 8-bit range and intensity image
    return img_range_intensity


## Crop point cloud within a pre-defined detection cube in space
def crop_pcl(lidar_pcl, configs):
    """ Crop point cloud within a pre-defined detection cube in 3D space as defined in configs
        and remove points with too low reflectivity.

    Args:
        - lidar_pcl (nd.array) : lidar point cloud
        - configs (easydict.EasyDict) : configuration data

    Returns:
        - lidar_pcl_crop (nd.array) : lidar point cloud cropped within a pre-defined detection cube
    """

    # remove points outside the detection cube defined in 'configs.lim_*'
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl_crop = lidar_pcl[mask]

    # remove point with too low reflectivity as defined in 'configs.lim_r'
    mask = np.where((lidar_pcl[:, 3] >= configs.lim_r[0]))

    # return lidar point cloud cropped within a pre-defined detection cube
    return lidar_pcl_crop


# create birds-eye view (bev) of lidar data
def bev_from_pcl(lidar_pcl, configs, visualize=True, pause_time_ms=0):
    """ Create birds-eye view of lidar point cloud data transforming it into a 2D color image
        where the first channel represents lidar intensity, the second channel represents the
        object height and the third channel represents the projected point density within the
        respective 2D bev image grid cell.

    Args:
        - lidar_pcl (nd.array) : 3D lidar point cloud cropped from a pre-configured detection cube
        - configs (dict) : configuration data
        - visualize (bool) : optional flag to turn visualization on if True or off if False

    Returns:
        - input_bev_maps (nd.array): lidar birds-eye view image as input to a lidar object detector
    """

    # remove lidar points outside a pre-configured detection cube and with too low reflectivity
    lidar_pcl_crop = crop_pcl(lidar_pcl, configs)

    ## Convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######
    #######
    print("student task ID_S2_EX1 (bev_from_pcl)")

    ## Step 1 : Compute bev-map discretization by dividing x-range by the bev-image height
    ##          (see configs)
    bev_map_discret = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height

    ## Step 2 : Create a copy of the lidar pcl and transform all metric x-coordinates into
    ##          bev-image coordinates
    lidar_pcl_cpy = np.copy(lidar_pcl_crop)
    lidar_pcl_cpy[:, 0] = np.int_(
        np.floor(lidar_pcl_cpy[:, 0] / bev_map_discret)) # lateral axis

    ## Step 3 : Perform the same operation as in step 2 for the metric y-coordinates, but
    ##          make sure that no negative bev-coordinates occur by centering the forward-
    ##          facing x-axis on the middle of the image
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_map_discret) + \
        (configs.bev_width + 1) / 2) # longitudinal axis

    ## Step 4: Shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl_cpy[:, 2] = lidar_pcl_cpy[:, 2] - configs.lim_z[0]

    ## Step 5 : Visualize cropped lidar point cloud using the function show_pcl from a previous task
    if visualize:
        # Alternative 1:
        if "o3d_bev_pcl_vis" not in globals():
            global o3d_bev_pcl_vis
            o3d_bev_pcl_vis = PointCloudVisualizer(
                lidar_pcl_cpy, window_name='Show 2D birds-eye-view on cropped Lidar Point Cloud',
                width=1280, height=720, left=100, top=100, pause_time_ms=pause_time_ms)
        else:
            o3d_bev_pcl_vis.update(lidar_pcl_cpy)
        # Alternative 2:
        #show_pcl(
        #    lidar_pcl_cpy,
        #    window_name='Show 2D birds-eye-view on Lidar Point Cloud',
        #    pause_time_ms=pause_time_ms
        #)
    #######
    ####### ID_S2_EX1 END #######

    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######
    #######
    print("student task ID_S2_EX2 (bev_from_pcl)")

    ## Step 1 : Create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))

    ## Step 2 : Re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then by
    ##          decreasing z resp. by -z (use numpy.lexsort). Make sure to invert z as sorting
    ##          is performed in ascending order and we want the top-most point for each cell.
    lidar_pcl_cpy[lidar_pcl_cpy[:, 3] > 1.0, 3] = 1.0  # clip (invalid) intensity values > 1.0
    height_idx = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_top = lidar_pcl_cpy[height_idx]

    ## Step 3 : Extract all points with identical x and y such that only the top-most z-coordinate
    ##          is kept (use numpy.unique), also, store the number of points per x,y-cell in a
    ##          variable named "counts" for use in the next task to compute the density layer of
    ##          the BEV map.
    _, unique_pos_idx, counts = np.unique(
        lidar_pcl_top[:, 0:2], axis=0, return_index=True, return_counts=True
    )
    lidar_pcl_top = lidar_pcl_top[unique_pos_idx]

    ## Step 4 : Assign the intensity value of each unique entry in lidar_pcl_top to the intensity
    ##          map make sure that the intensity is scaled in such a way that objects of interest
    ##          (e.g. vehicles) are clearly visible, also, make sure that the influence of outliers
    ##          is mitigated by normalizing intensity on the difference between the max. and min.
    ##          value within the point cloud
    # normalize intensity map over the range of 1...99 percentile of the intensity spectrum
    lower_percentile = np.percentile(lidar_pcl_top[:, 3], configs.intensity_lower_percentile)
    upper_percentile = np.percentile(lidar_pcl_top[:, 3], configs.intensity_upper_percentile)
    intensity_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = (
        np.clip(
            lidar_pcl_top[:, 3], a_min=lower_percentile, a_max=upper_percentile
        ) - lower_percentile) / (upper_percentile - lower_percentile)
    # alternatively: normalize intensity map over half the maximum of the intensity spectrum
    #intensity_map = np.amax(lidar_pcl_top[:, 3])/2 * lidar_pcl_top[:, 3] / \
    #    (np.amax(lidar_pcl_top[:, 3]) - np.amin(lidar_pcl_top[:, 3]))

    ## Step 5 : Temporarily visualize the intensity map using OpenCV to make sure that vehicles
    ##          separate well from the background
    if visualize:
        intensity_image = (255 * intensity_map).astype(np.uint8) # normalized 8-bit intensity map
        cv2.imshow('Lidar point cloud normalized intensity map image (8 bit)', intensity_image)
        cv2.waitKey(pause_time_ms)
        cv2.destroyAllWindows()
    #######
    ####### ID_S2_EX2 END #######


    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######
    #######
    print("student task ID_S2_EX3 (bev_from_pcl)")

    ## Step 1 : Create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))

    ## Step 2 : Assign the height value of each unique entry in lidar_pcl_top to the height map
    ##          make sure that each entry is normalized on the difference between the upper and
    ##          lower height defined in the config file use the lidar_pcl_top data structure
    ##          from the previous task to access the pixels of the height_map.
    height_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 2] / \
        np.float32(configs.lim_z[1] - configs.lim_z[0]) # normalized height map

    ## Step 3 : Temporarily visualize the intensity map using OpenCV to make sure that vehicles
    ##          separate well from the background
    if visualize:
        height_image = (255 * height_map).astype(np.uint8) # normalized 8-bit height map
        cv2.imshow('Lidar point cloud normalized height map image (8 bit)', height_image)
        cv2.waitKey(pause_time_ms)
        cv2.destroyAllWindows()
    #######
    ####### ID_S2_EX3 END #######

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    #_, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalized_counts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalized_counts

    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map by a batch size dimension before converting into a tensor
    s_1, s_2, s_3 = bev_map.shape
    bev_maps = np.zeros((1, s_1, s_2, s_3))
    bev_maps[0] = bev_map

    # convert birds-eye view map into a tensor
    bev_maps = torch.from_numpy(bev_maps)

    # move input bev tensor to gpu and return it
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps
