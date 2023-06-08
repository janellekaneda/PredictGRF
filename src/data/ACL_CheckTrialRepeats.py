# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 02:20:01 2022

@author: jkaneda

Checks for duplicated trials in the ACL dataset,
and also that only plates 1 and 2 (double jump) or only plate 2 (all other tasks)
were used.
"""

#%% IMPORTS

import numpy as np
import os

#%% BATCH PROCESS

# # PARAMETERS # #
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload' # your base directory
FEAT_TO_CHECK = 18 # Index number of feature to check
ROUND = True # Toggle if want to round force plate data to no decimals for checking
# # # # # # # # #

datadir = os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices')

# Get subject IDs.
subjs = os.listdir(datadir)
subjs.remove('.DS_Store')

#subjs = ['102913_315']

# Loop through subjects.
for subj in subjs:
    
    # Init a list to store all test arrays.
    feat_across_trials = []
    # Loop over trials.
    feat_mats = os.listdir(os.path.join(datadir, subj)) # file names
    
    for feat_mat in feat_mats:
        # Load in feature matrix.
        data = np.load(os.path.join(datadir, subj, feat_mat), allow_pickle=True)
        feat_across_trials.append(data[:, FEAT_TO_CHECK])
        
        # Check that correct force plates were used.
        grf_mag_r = data[:,42]
        grf_mag_l = data[:,46]
        if ROUND:
            grf_mag_r = np.around(grf_mag_r, decimals=0)
            grf_mag_l = np.around(grf_mag_l, decimals=0)
            
        num_timesteps = data.shape[0]
        if "_dj" in feat_mat.lower(): # both legs should have signal for double jump tasks
            if np.count_nonzero(grf_mag_r) < int(num_timesteps * 0.25) or np.count_nonzero(grf_mag_l) < int(num_timesteps * 0.25):
                print(f"Error for Subject {subj}, Trial {feat_mat}:")
                print(f"Num nonzero grf_mag_r: {np.count_nonzero(grf_mag_r)}")
                print(f"Num nonzero grf_mag_l: {np.count_nonzero(grf_mag_l)}")
                print(f"Num timesteps: {num_timesteps}")
                
        else: # exactly one leg should have no signal for single leg tasks
            if (np.count_nonzero(grf_mag_r) < int(num_timesteps * 0.25) and np.count_nonzero(grf_mag_l) < int(num_timesteps * 0.25)) or (np.count_nonzero(grf_mag_r) > int(num_timesteps * 0.25) and np.count_nonzero(grf_mag_l) > int(num_timesteps * 0.25)):
                print(f"Error for Subject {subj}, Trial {feat_mat}:")
                print(f"Num nonzero grf_mag_r: {np.count_nonzero(grf_mag_r)}")
                print(f"Num nonzero grf_mag_l: {np.count_nonzero(grf_mag_l)}")
                print(f"Num timesteps: {num_timesteps}")
                
    # Get data arrays for unique trials.
    unique_trials = set(tuple(arr) for arr in feat_across_trials)
    
    # Return subject ID if detect duplicated trials.
    if len(unique_trials) != len(feat_mats):
        print(f"Subject {subj} has duplicated trials:")
        print(f"Number of unique trials: {len(unique_trials)}. Number of feat mats: {len(feat_mats)}.")