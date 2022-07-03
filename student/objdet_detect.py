# -------------------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
# Copyright (C) 2022, Andreas Albrecht (modifications on starter code)
#
# Purpose of this file : Detect 3D objects in lidar point clouds using deep learning
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# -------------------------------------------------------------------------------
#

# general package imports
import os
import sys
import numpy as np
import torch
from easydict import EasyDict as edict

# add project directory to python path to enable relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# model-related
from tools.objdet_models.resnet.models import fpn_resnet
from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing
from tools.objdet_models.resnet.utils.torch_utils import _sigmoid

from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2


# load model-related parameters into an edict
def load_configs_model(model_name='darknet', configs=None):
    """ Load object detection model parameters into an edict dictionary.

    Args:
        - model_name (str) : name of the object detection model. Defaults to 'darknet'.
        - configs (edict) : model configuration data dictionary. Defaults to None.
    """

    # init config file, if none has been passed
    if configs is None:
        configs = edict()

    # get parent directory of this file to enable relative paths
    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))

    # set parameters according to model type
    if model_name == "darknet":
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'darknet')
        configs.pretrained_filename = os.path.join(
            configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth'
        )
        configs.arch = 'darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.5
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False

    elif model_name == 'fpn_resnet':
        ####### ID_S3_EX1-3 START #######
        #######
        print("student task ID_S3_EX1-3")
        # reference: https://github.com/maudzung/SFA3D/blob/master/sfa/test.py
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'resnet')
        configs.pretrained_filename = os.path.join(
            configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth'
        )
        configs.arch = 'fpn_resnet_18'
        configs.batch_size = 4
        configs.K = 50

        configs.conf_thresh = 0.5
        configs.nms_thresh = 0.4

        configs.num_samples = None
        configs.num_workers = 4 # defaults to 1
        configs.output_format = 'image' # or 'video'

        configs.distributed = False  # For testing on 1 GPU only
        configs.pin_memory = True

        configs.input_size = (608, 608)
        configs.hm_size = (152, 152)
        configs.down_ratio = 4
        configs.max_objects = 50

        configs.imagenet_pretrained = False
        configs.head_conv = 64
        configs.num_classes = 3
        configs.num_center_offset = 2
        configs.num_direction = 2  # sin, cos
        configs.num_z = 1
        configs.num_dim = 3

        configs.heads = {
            'hm_cen': configs.num_classes,
            'cen_offset': configs.num_center_offset,
            'direction': configs.num_direction,
            'z_coor': configs.num_z,
            'dim': configs.num_dim
        }
        configs.num_input_features = 4
        #######
        ####### ID_S3_EX1-3 END #######

    else:
        raise ValueError("Error: Invalid model name")

    # GPU vs. CPU
    configs.no_cuda = True # if true, cuda is not used
    configs.gpu_idx = 0  # GPU index to use.
    configs.device = torch.device('cpu' if configs.no_cuda else f'cuda:{configs.gpu_idx}')

    return configs


# load all object-detection parameters into an edict
def load_configs(model_name='fpn_resnet_18', configs=None):
    """ Load object detection model parameters into an edict dictionary.

    Args:
        - model_name (str) : name of the object detection model. Defaults to 'fpn_resnet_18'.
        - configs (edict) : model configuration data dictionary. Defaults to None.
    """

    # init config file, if none has been passed
    if configs is None:
        configs = edict()

    # birds-eye view (bev) parameters
    configs.lim_x = [0, 50] # detection range in m
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0] # reflected lidar intensity
    configs.bev_width = 608  # pixel resolution of bev image
    configs.bev_height = 608

    # set intensity map normalization parameters
    configs.intensity_lower_percentile = 1
    configs.intensity_upper_percentile = 99

    # add model-dependent parameters
    configs = load_configs_model(model_name, configs)

    # set minimum IoU threshold for true positive detections
    configs.iou_thresh = 0.5

    # visualization parameters
    # width of result image (height may vary)
    configs.output_width = 608
    # color settings for all object classes: 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]

    return configs


# create model according to selected model type
def create_model(configs):
    """ Create object detection model according to the selected model type and model
        configuration parameters.

    Args:
        - configs (edict) : model configuration data dictionary. Defaults to None.
    """

    # check for availability of model file
    assert os.path.isfile(configs.pretrained_filename), f'No file at {configs.pretrained_filename}'

    # create model depending on architecture name
    if (configs.arch == 'darknet') and (configs.cfgfile is not None):
        print('using darknet')
        # create model acccording to the configuration
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)

    elif 'fpn_resnet' in configs.arch:
        print('using ResNet architecture with feature pyramid network (fpn)')

        ####### ID_S3_EX1-4 START #######
        #######
        print("student task ID_S3_EX1-4")
        # reference: https://github.com/maudzung/SFA3D/blob/master/sfa/models/model_utils.py
        try:
            # get number of layers from configured architecture
            arch_parts = configs.arch.split('_')
            num_layers = int(arch_parts[-1])
        except:
            raise ValueError
        # create model acccording to the configuration
        model = fpn_resnet.get_pose_net(
            num_layers=num_layers, heads=configs.heads, head_conv=configs.head_conv,
            imagenet_pretrained=configs.imagenet_pretrained
        )
        #######
        ####### ID_S3_EX1-4 END #######

    else:
        assert False, 'Undefined model backbone'

    # load model weights
    model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
    print(f'Loaded weights from {configs.pretrained_filename}\n')

    # set model to evaluation state
    configs.device = torch.device('cpu' if configs.no_cuda else f'cuda:{configs.gpu_idx}')
    model = model.to(device=configs.device)  # load model to either cpu or gpu
    model.eval() # set model to evaluation mode

    return model


# detect trained objects in birds-eye view
def detect_objects(input_bev_maps, model, configs):
    """ Detect objects of trained classes in a 2D birds-eye-view on a 3D lidar point cloud
        projected to the drivable ground.

    Args:
        - input_bev_maps (tensor) : birds-eye-view map of the 3D lidar point cloud projected
            to the drivable ground as input tensor to the object detection model of shape
            (batch size, color channels, width, height)
        - configs (edict) : model configuration data dictionary. Defaults to None.
    """

    # deactivate autograd engine during test to reduce memory usage and speed up computations
    with torch.no_grad():

        # perform inference
        outputs = model(input_bev_maps)

        # decode model output into target object format
        if 'darknet' in configs.arch:

            # perform post-processing
            output_post = post_processing_v2(
                outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh
            )
            # loop over list of post processed object detection tensors
            detections = []
            for detection in output_post:
                if detection is not None:
                    detection = detection.cpu().numpy().astype(np.float32)
                    for obj in detection:
                        _x, _y, _w, _l, _im, _re, _, _, _ = obj
                        _z = 0.0
                        _h = 1.50
                        _class_id = 1 # only object class "car" considered
                        _yaw = np.arctan2(_im, _re)
                        detections.append([_class_id, _x, _y, _z, _h, _w, _l, _yaw])

        elif 'fpn_resnet' in configs.arch:
            # decode output and perform post-processing

            ####### ID_S3_EX1-5 START #######
            #######
            print("student task ID_S3_EX1-5")
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            # detections size: (batch_size, K, 10)
            # (scores x 1, ys x 1, xs x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
            # (scores-0:1, ys-1:2, xs-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
            detections = decode(
                outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                outputs['dim'], K=configs.K
            )
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs)
            detections = detections[0][1] # object detections from first batch
            #######
            ####### ID_S3_EX1-5 END #######


    ####### ID_S3_EX2 START #######
    #######
    # Extract 3d bounding boxes from model response
    print("student task ID_S3_EX2")
    objects = []

    ## step 1 : check whether there are any detections
    print(f'Number of detected objects: {len(detections)}')
    if len(detections) > 0:

        # get bev boundaries
        bev_bound_size_x = configs.lim_x[1] - configs.lim_x[0]
        bev_bound_size_y = configs.lim_y[1] - configs.lim_y[0]

        ## step 2 : loop over all detections
        for det in detections:

            # extract details from current object detection
            # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
            # _score, _x, _y, _z, _h, _w, _l, _yaw = det
            _, _x, _y, _z, _h, _w, _l, _yaw = det

            ## step 3 : perform conversion using the limits for x, y and z set in configs structure
            obj_heading = -_yaw
            obj_center_x = _y / configs.bev_height * bev_bound_size_x + configs.lim_x[0]
            obj_center_y = _x / configs.bev_width * bev_bound_size_y + configs.lim_y[0]
            obj_center_z = _z + configs.lim_z[0]
            obj_height = _h
            obj_width = _w / configs.bev_width * bev_bound_size_y
            obj_length = _l / configs.bev_height * bev_bound_size_x

            # set object class id to object class "car" (obj_class_id = 1)
            obj_class_id = 1 # only object class "car" considered

            # create object with converted properties
            obj = [
                obj_class_id, obj_center_x, obj_center_y, obj_center_z,
                obj_height, obj_width, obj_length, obj_heading
            ]

            ## step 4 : append the current object to the 'objects' array
            objects.append(obj)
    #######
    ####### ID_S3_EX2 START #######

    return objects
