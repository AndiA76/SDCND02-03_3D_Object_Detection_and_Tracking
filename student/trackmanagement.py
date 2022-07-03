# -------------------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
# Copyright (C) 2022, Andreas Albrecht (modifications on starter code)
#
# Purpose of this file : Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# -------------------------------------------------------------------------------
#

# general package imports
import os
import sys
import collections
import numpy as np

# add project directory to python path to enable relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# project-specific imports
import misc.params as params 

class Track:
    '''Track class with state, covariance, id, score'''
    def __init__(self, meas, id):
        ''' Initialize new object track instance with a dedicated track id
        Args:
            - meas (class measurements.measurement) : measurement holding the measurement
                vector z from lidar or camera sensor and the respective measurement noise
                covariance matrix R
            - id (int) : track id
        '''
        print('creating track no.', id)
        # Store track id
        self.id = id

        ############
        # Step 2: initialization:
        # - replace fixed track initialization values by initialization of x and P based
        #   on unassigned measurement transformed from sensor to vehicle coordinates
        # - initialize track state and track score with appropriate values
        ############

        # Get new position measurement in homogeneous coordinates
        pos_sens = np.ones((4, 1))
        pos_sens[0:3] = meas.z[0:3]

        # Transform position measurement to vehicle coordinates using homogenesous coordinates
        pos_veh = meas.sensor.sens_to_veh * pos_sens

        # Save initial state x from position measurement assuming velocity to be zero
        self.x = np.zeros((6, 1))
        self.x[0:3] = pos_veh[0:3]

        # Get rotation matrix from sensor to vehicle coordinates transform
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3]

        # Set up position estimation error covariance
        P_pos = M_rot * meas.R * np.transpose(M_rot)

        # Set up velocity estimation error covariance assuming larger uncertainties for unknown velocity compontents
        P_vel = np.matrix([
            [params.sigma_p44**2, 0, 0],
            [0, params.sigma_p55**2, 0],
            [0, 0, params.sigma_p66**2]
        ])

        # Initialize overall state estimation error covariance matrix P
        self.P = np.zeros((6, 6))
        self.P[0:3, 0:3] = P_pos
        self.P[3:6, 3:6] = P_vel

        # Set track state to 'initialized'
        self.state = 'initialized'

        # Set minimal initial track score with respect to the tracking window size
        self.score = 1./params.window

        ############
        # END student code
        ############

        # Initialize estimated target object dimensions as further track attributes
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        # Initialize estimated target object heading, or yaw angle, respectively
        self.yaw =  np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw))
        # Initialize measurement time as further attribute of the current track
        self.t = meas.t

    def set_x(self, x):
        ''' Set estimated state vector 
        Args:
            - x (np.ndarray) : 6 x 1 state vector x
        '''
        self.x = x

    def set_P(self, P):
        ''' Set state estimation error covariance matrix P 
        Args:
            - x (np.ndarray) : 6 x 6 state estimation error covariance matrix P
        '''
        self.P = P

    def set_t(self, t):
        ''' Set measurement time stamp
        Args:
            - t (float) : measurement time stamp
        '''
        self.t = t

    def update_attributes(self, meas):
        ''' Update track attributes 
        Args:
            - meas (measurement.Measurement class) : Current measurement
        '''
        # Use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == 'lidar':
            # Get sliding average parameter for dimension estimation
            c = params.weight_dim
            # Update estimated target object dimensions
            self.width = c*meas.width + (1 - c)*self.width
            self.length = c*meas.length + (1 - c)*self.length
            self.height = c*meas.height + (1 - c)*self.height
            # Get rotation matrix from sensor to vehicle coordinates transform
            M_rot = meas.sensor.sens_to_veh[0:3, 0:3]
            # Update estimated target object heading, or yaw angle, respectively
            self.yaw = np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw))


###################

class Trackmanagement:
    ''' Track manager with logic for initializing and deleting objects '''
    def __init__(self):
        self.N = 0 # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []

    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):
        ''' Manage tracks
        Args:
            - unassigned_tracks (list) : list of unassigned tracks
            - unassigned_meas (list) : list of unassigned measurements
            - meas_list (list) : list of measurements
        '''
        ############
        # Step 2: Implement track management:
        # - decrease the track score for unassigned tracks
        # - delete tracks if the score is too low or P is too big (check params.py for parameters
        #   that might be helpful, but feel free to define your own parameters)
        ############

        # Decrease score for unassigned tracks
        for i in unassigned_tracks:
            print(f'unassigned_track no. {i}')
            track = self.track_list[i]
            # Check visibility
            if meas_list: # if not empty
                print('Measurements exist')
                if meas_list[0].sensor.in_fov(track.x):
                    print('Decrease score for unassigned tracks')
                    # Decrease track score by one increment
                    track.score -= 1 / params.window

        # Delete old confirmed tracks if track score falls below deletion threshold,
        # or delete initialized / tentative tracks if state estimation covariance
        # (uncertainty) grows too big or if track score is too low, respectively
        for track in self.track_list:
            # Init delete track flag to False
            delete_track = False
            if track.state == 'confirmed' and \
                track.score < params.delete_threshold:
                # Delete confirmed track if track.score falls below delete threshold
                delete_track = True
            elif track.state in ['initialized', 'tentative'] and \
                track.score < 1 / params.window:
                # Delete initialized or tentative track if track.score is smaller than 1 increment
                delete_track = True
            if track.P[0, 0] > params.max_P or track.P[1, 1] > params.max_P:
                delete_track = True
            if delete_track:
                self.delete_track(track)

        ############
        # END student code
        ############

        # Initialize new track with unassigned measurement
        for j in unassigned_meas:
            # Only initialize with lidar measurements
            if meas_list[j].sensor.name == 'lidar':
                self.init_track(meas_list[j])

    def add_track_to_list(self, track):
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        track = Track(meas, self.last_id + 1)
        self.add_track_to_list(track)

    def delete_track(self, track):
        print('deleting track no.', track.id)
        self.track_list.remove(track)

    def handle_updated_track(self, track):
        ############
        # Step 2: Implement track management for updated tracks:
        # - increase track score
        # - set track state to 'tentative' or 'confirmed'
        ############

        # Increase track score (saturating at 1.0 or 100% estimated existance probability)
        if track.score + 1 / params.window > 1.0:
            track.score = 1.0
        else:
            track.score += 1 / params.window

        # Update track state if track score passes certain thresholds
        if track.score > params.confirmed_threshold:
            # Set track state to 'confirmed' if track.score is larger than confirmed threshold
            track.state = 'confirmed'
        elif track.score > 1 / params.window:
            # Set track state to 'tentative' if track.score is larger than one minimum increment
            track.state = 'tentative'

        ############
        # END student code
        ############ 