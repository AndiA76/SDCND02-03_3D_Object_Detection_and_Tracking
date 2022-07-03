# -------------------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
# Copyright (C) 2022, Andreas Albrecht (modifications on starter code)
#
# Purpose of this file : Kalman filter class
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

# add project directory to python path to enable relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# import project specific tracking parameters
import misc.params as params


## Extended Kalman filter class
class Filter:
    '''Extended Kalman filter class with a constant velocity model in 3D space'''
    def __init__(self):
        ## Process model parameters
        # process model dimension for a constant velocity model in 3D space
        # state vector: [p_x, p_y, p_z, v_x, v_y, v_z] => dim_state = 6
        self.dim_state = params.dim_state

        ## Kalman filter parameters
        # time step
        self.dt = params.dt
        # process noise variable for process noise covariance matrix Q
        self.q = params.q

    def F(self):
        ''' System matrix of the Extended Kalaman Filter (EKF) for a constant
            velocity vehicle model in 3D space as inner model.
        Returns:
            - F (6x6 np.ndarray) : system matrix
        '''
        ############
        ## Step 1: Implement and return system matrix F
        ############

        # get time step increment from this Kalman filter instance
        dt = self.dt

        # calculate and return system matrix F
        return np.matrix(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ]
        )

        ############
        # END student code
        ############

    def Q(self):
        ''' Process noise covariance matrix of the Extended Kalman Filter (EKF)
        Returns:
            - Q (6x6 np.ndarray) : process noise covariance matrix
        '''
        ############
        ## Step 1: Implement and return process noise covariance Q
        ############

        # get time step increment from this Kalman filter instance
        dt = self.dt

        # get process noise covariance matrix scale factor from this Kalman filter instance
        q = self.q

        # set process noise covariance matrix coefficients
        q1 = ((dt**3)/3) * q
        q2 = ((dt**2)/2) * q
        q3 = dt * q

        # calculate and return process noise covariance matrix Q
        return np.matrix(
            [
                [q1, 0, 0, q2, 0, 0],
                [0, q1, 0, 0, q2, 0],
                [0, 0, q1, 0, 0, q2],
                [q2, 0, 0, q3, 0, 0],
                [0, q2, 0, 0, q3, 0],
                [0, 0, q2, 0, 0, q3]
            ]
        )

        ############
        # END student code
        ############

    def predict(self, track):
        ''' Prediction step of the Extended Kalman Filter (EKF)
        Args:
            - track (class trackmanagement.track) : object track holding the estimates of the state
                vector x and the process noise covariance matrix P of the respective object
        '''
        ############
        ## Step 1: Predict the state vector x and the estimation error covariance matrix P for next
        ##         timestep t + dt, save x and P in object track
        ############

        # get state vector x of the current time step from object track
        x = track.x

        # get estimation error covariance matrix P of the current time step from object track
        P = track.P

        # calculate system matrix F from this Kalman filter instance
        F = self.F()

        # calcualate process noise covariance matrix Q from this Kalman filtern instance
        Q = self.Q()

        # predict state vector x for next timestep t + dt
        x = F*x

        # predict estimation error covariance matrix P for next timestep t + dt
        P = F*P*F.transpose() + Q

        # save predicted state vector in object track
        track.set_x(x)

        # save predicted estimation error covariance matrix in object track
        track.set_P(P)

        ############
        # END student code
        ############

    def update(self, track, meas):
        ''' Update step of the Extended Kalman Filter (EKF).
        Args:
            - track (class trackmanagement.track) : object track holding the estimates of the state
                vector x and the process noise covariance matrix P of the respective object.
            - meas (class measurements.measurement) : measurement holding the measurement vector z
                from lidar or camera sensor and the respective measurement noise covariance matrix R
        '''
        ############
        ## Step 1: Update state vector x and process noice covariance matrix P with an
        ##         associated sensor measurement, save x and P in the current track
        ############

        ## Update state and covariance with associated sensor measurement
        # get current predicted state vector

        # get state vector x of the current time step from object track
        x = track.x

        # get estimation error covariance matrix P of the current time step from object track
        P = track.P

        # get Jacobian H of the non-/linear sensor measurement function for the current state x
        H = meas.sensor.get_H(x)

        # calculate residual gamma between sensor measuremnt and predicted measurement
        gamma = self.gamma(track, meas)

        # calculate residual covariance matrix S
        S = self.S(track, meas, H)

        # calculate Kalman filter gain matrix
        K = P*H.transpose()*np.linalg.inv(S)

        # measurement update of the state vector
        x = x + K*gamma

        # get identity matrix fit to state dimension
        I = np.identity(self.dim_state)

        # measurement update of the process noise covariance matrix
        P = (I - K*H) * P

        # save measurement update of the state vector in current track
        track.set_x(x)

        # save measurement update of the estimation error covariance matrix in current track
        track.set_P(P)

        ############
        # END student code
        ############
        # save sensor measurement in the current track
        track.update_attributes(meas)

    def gamma(self, track, meas):
        ''' Calculate the residual gamma between sensor measurement and predicted measurement of the
            Extended Kalman Filter (EKF).
        Args:
            - track (class trackmanagement.track) : object track holding the estimates of the state
                vector x and the process noise covariance matrix P of the respective object.
            - meas (class measurements.measurement) : measurement holding the measurement vector z
                from lidar or camera sensor and the respective measurement noise covariance matrix R
        '''
        ############
        ## Step 1: Calculate and return residual gamma between sensor measurement and
        ##         predicted measurement
        ############

        # get new sensor measurement
        z = meas.z

        # get state vector x of the current time step from object track
        x = track.x

        # evaluate the non-linear sensor measurement function at current state x
        hx = meas.sensor.get_hx(x)

        # calculate and return residual gamma between sensor measurement and predicted
        # measurement using a non-linear sensor model and measurement function hx(x)
        return z - hx

        ############
        # END student code
        ############

    def S(self, track, meas, H):
        ''' Calculate the residual error covariance matrix S of the Extended Kalman Filter (EKF)
        Args:
            - track (class trackmanagement.track) : object track holding the estimates of the state
                vector x and the process noise covariance matrix P of the respective object.
            - meas (class measurements.measurement) : measurement holding the measurement vector z
                from lidar or camera sensor and the respective measurement noise covariance matrix R
            - H (np.ndarray): 3x3 measurement matrix for lidar or 2x3 Jacobian of the non-linear
                measurement function h(x) for camera
        '''
        ############
        ## Step 1: Calculate and return residual error covariance matrix S
        ############

        # get estimation error covariance matrix P of the current time step from object track
        P = track.P

        # get measurement noise covariance matrix R from sensor measurement
        R = meas.R

        # calculate and return residual error covariance matrix S
        return H*P*H.transpose() + R

        ############
        # END student code
        ############ 