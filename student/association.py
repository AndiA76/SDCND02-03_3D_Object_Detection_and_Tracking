# -------------------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
# Copyright (C) 2022, Andreas Albrecht (modifications on starter code)
#
# Purpose of this file : Data association class with single nearest neighbor
# association and gating based on Mahalanobis distance
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
from munkres import Munkres, print_matrix
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# project-specific imports
import misc.params as params

class Association:
    ''' Data association class with single or global nearest neighbor association and
        gating based on Mahalanobis distance
    '''
    def __init__(self, association_method='SNN'):
        ''' Association class constructor
        Args:
            - association_method (str) : track-measurement assocation method (options:
                'SNN', 'GNN'). Defaults to 'SNN'.
        '''
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        self.association_method = association_method

    def associate(self, track_list, meas_list, KF):
        ''' Associate tracks and measurements with the smallest Mahalanobis distance,
            which must be smaller than the gating limit.
        Args:
            - track_list (list) : track list
            - meas_list (list) : measurement list
            - KF (class Filter) Kalman filter
        '''
        ############
        # Step 3: Association:
        # - replace association_matrix with the actual association matrix based on
        #   Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############

        # Reset lists for unassigned tracks and measurements
        self.unassigned_tracks = []
        self.unassigned_meas = []

        # Reset association matrix
        self.association_matrix = np.matrix([])

        # Get number of tracks N (=number of association matrix rows)
        num_tracks = len(track_list)
        # Get number of measurements M (= number of association matrix columns)
        num_meas = len(meas_list)

        # Set up lists of unassigned tracks and measurements starting containing ascending indices
        # for all existing tracks and measurements that need to be associated with one another
        if num_tracks > 0:
            self.unassigned_tracks = list(range(num_tracks))
        if num_meas > 0:
            self.unassigned_meas = list(range(num_meas))

        # Set up association matrix if there are tracks and measurements
        if num_tracks > 0 and num_meas > 0:
            if self.association_method == 'GNN':
                # Initialize N x M association matrix with max int values
                self.association_matrix = sys.maxsize * np.ones((num_tracks, num_meas))
            else:
                # Initialize N x M association matrix with inf values
                self.association_matrix = np.inf * np.ones((num_tracks, num_meas))

            # Loop over all tracks and all measurements to populate the association matrix
            for i in range(num_tracks):
                track = track_list[i]
                for j in range(num_meas):
                    meas = meas_list[j]
                    dist = self.MHD(track, meas, KF)
                    # Check if Mahalanobis distance is smaller than the gating limit
                    if self.gating(dist, meas.sensor):
                        # Update association matrix
                        self.association_matrix[i,j] = dist

        ############
        # END student code
        ############

    def get_globally_closest_tracks_and_meas(self):
        ''' Find the globally closest track and measurement pairs with respect to the sum of
            their Mahalanobis distances, remove the respective indices from the unassigned
            track and measurement lists.
        Returns:
            - update_indices (list) : list of globally closest track-measurement pairs
        '''

        # Init list of associated track and measurement indexes as empty list
        update_indexes = []

        # Get the current association matrix
        A = self.association_matrix

        # Check if there are any potential tracks or measurements to be associated
        if A.shape[0]>0 and A.shape[1]>0 and np.any(A) and np.min(A) < sys.maxsize:
            # Munkres / Hungarian algorithm:
            # https://pypi.org/project/munkres3/
            # https://github.com/datapublica/munkres/blob/master/munkres.py

            # Create Munkres() object (Munkres expects matrices given as lists)
            munkres_obj = Munkres()

            # Compute association indexes with minimum global cost w.r.t. MHD sum
            indexes = munkres_obj.compute(A.tolist())

            # Update associated tracks with measurements
            #print_matrix(A.tolist(), msg='Lowest cost through this matrix:')

            # Init total cost (only needed for printing intermediate results)
            #total = 0

            # Get associated track index (row) and measurement index (column)
            for idx_track, idx_meas in indexes:

                # Get Mahalanobis distance of this track measurement pair
                value = A[idx_track][idx_meas]

                # Check if the initial value in the association matrix has been changed
                if value == sys.maxsize:
                    # continue if value in association matrix has not been any update
                    continue
                # Update total cost w.r.t. MHD
                #total += value

                # Add track measurement pair to update list
                update_indexes.append((idx_track, idx_meas))
                #print(f'({idx_track}, {idx_meas}) -> {value}')

                # Remove this track and measurement from the unassigned track and measurement lists
                self.unassigned_tracks.remove(idx_track)
                self.unassigned_meas.remove(idx_meas)
            
            # Show overall cost w.r.t. MHD of the associated track and measurement constellation
            #print(f'total association cost (w.r.t. MHD): {total}')

        # Return update list of associated track and measurement indexes
        return update_indexes

    def get_closest_track_and_meas(self):
        ''' Find the closest track and measurement with respect to their Mahalanobis distance,
            remove the respective indices from the association matrix and the unassigned track
            and measurement lists.
        Returns:
            - update_track (int) : closest track
            - update_meas (int) : closest measurement
        '''
        ############
        # Step 3: Find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # Find closest track and measurement for next update
        A = self.association_matrix
        # Check if there are any tracks or measurements, resp. track-measurement associations
        if not np.any(A) or np.min(A) == np.inf:
            # Return nan
            return np.nan, np.nan

        # Get the indices of the track and measurement pair with the smallest Mahalanobis distance
        ij_min = np.unravel_index(np.argmin(A, axis=None), A.shape)
        idx_track = ij_min[0]
        idx_meas = ij_min[1]

        # Update association matrix by deleting the row (track) and column (measurement) with the
        # smallest Mahalanobis distance
        A = np.delete(A, idx_track, 0)
        A = np.delete(A, idx_meas, 1)
        self.association_matrix = A

        # Update this track with this measurement
        update_track = self.unassigned_tracks[idx_track]
        update_meas = self.unassigned_meas[idx_meas]

        # Remove this track and measurement from the unassigned track and measurement lists
        self.unassigned_tracks.remove(update_track)
        self.unassigned_meas.remove(update_meas)

        ############
        # END student code
        ############
        return update_track, update_meas

    def gating(self, MHD, sensor):
        ''' Gating function to prevent association of object tracks with measurements
            ouside gating limits.
        Args:
            - MHD (float) : Mahalanobis distance
            - sensor (class Sensor) : sensor object containing transformation matrices
                from sensor to vehicle coordinates
        Returns:
            - inside_gate (bool): Return True if measurement lies inside gate, otherwise False
        '''
        ############
        # Step 3: Return True if measurement lies inside gate, otherwise False
        ############

        # Calculate the gate limit
        limit = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)

        # Check if the Mahalanobis distance between track and measurement lies inside the gate limit
        inside_gate = bool(MHD < limit)

        # Return True if measurement lies inside the gating limit
        return inside_gate

        ############
        # END student code
        ############

    def MHD(self, track, meas, KF):
        ''' Calculate Mahalanobis distance between a track state and a measurement.
        Args:
            - track (class Track) : object track
            - meas (class Measurement) : sensor measurement
            - KF (class Filter) : Kalman filter
        '''
        ############
        # Step 3: calculate and return Mahalanobis distance
        ############

        # Get Jacobian H of the non-/linear sensor measurement function for the current state x
        H = meas.sensor.get_H(track.x)

        # Calculate residual gamma between sensor measuremnt and predicted measurement
        gamma = KF.gamma(track, meas)

        # Caculate residual error covariance matrix S
        S = KF.S(track, meas, H)

        # Calculate Mahalanobis (MHD) distance between this track and this measurement
        MHD = gamma.transpose()*np.linalg.inv(S)*gamma

        # Return MHD distance
        return MHD

        ############
        # END student code
        ############

    def associate_and_update(self, manager, meas_list, KF):
        ''' Associate measurements and tracks and update tracks with measurements.
        Args:
            - manager (class Trackmanagement) : object track manager
            - meas (class Measurement) : sensor measurement list
            - KF (class Filter) : Kalman filter
        '''

        # Associate measurements and tracks with the smallest Mahalanobis distance,
        # which must be smaller than the gating limit.
        self.associate(manager.track_list, meas_list, KF)

        # Use single nearest neighbor (SNN) association method
        if self.association_method == 'SNN':
            print('using single nearest neighbor association (SNN) ...')
            # Update associated tracks with measurements
            while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:

                # search for next association between a track and a measurement
                idx_track, idx_meas = self.get_closest_track_and_meas()
                if np.isnan(idx_track):
                    print('---no more associations---')
                    break
                track = manager.track_list[idx_track]

                # check visibility, only update tracks in fov
                if not meas_list[0].sensor.in_fov(track.x):
                    continue

                # Kalman update
                print(
                    'update track', track.id, 'with', meas_list[idx_meas].sensor.name,
                    'measurement', idx_meas
                )
                KF.update(track, meas_list[idx_meas])

                # update score and track state
                manager.handle_updated_track(track)

                # save updated track
                manager.track_list[idx_track] = track

        # Global nearest neighbor (GNN) association method => todo: check track management for correct incrementation of track scores
        elif self.association_method == 'GNN':
            print('using global nearest neighbor association (GNN) ...')
            # Find constellation of associated track measurement pairs with minimal overall sum of Mahalanobis distances
            update_indexes = self.get_globally_closest_tracks_and_meas()

            # Check if there are any associations of tracks and measurements
            if len(update_indexes) > 0:
                # Loop over all associated track measurement pairs
                for idx_track, idx_meas in update_indexes:
                    # Get current track from track list
                    track = manager.track_list[idx_track]

                    # Check visibility, only update tracks in fov
                    if not meas_list[0].sensor.in_fov(track.x):
                        continue

                    # Kalman update
                    print(
                        'update track', track.id, 'with', meas_list[idx_meas].sensor.name,
                        'measurement', idx_meas
                    )
                    KF.update(track, meas_list[idx_meas])

                    # Update score and track state
                    manager.handle_updated_track(track)

                    # Save updated track
                    manager.track_list[idx_track] = track
            else:
                print('---no associations---')
        else:
            print('Error: Association method not implemented')

        # run track management
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)

        for track in manager.track_list:
            print('track', track.id, 'score =', track.score)
