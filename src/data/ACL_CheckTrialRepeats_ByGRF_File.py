# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 02:20:01 2022

@author: jkaneda

Checks for duplicated trials in the ACL dataset,
and also that only plates 1 and 2 (double jump) or only plate 2 (all other tasks)
were used.

EDIT: Sat Jan 7 2023 -- change to check based on downsampled/trimmed GRF file,
since have to make edits via GRF file in build_features_utils.
"""

#%% IMPORTS

import numpy as np
import pandas as pd
import os
from opensim_sto_reader import readMotionFile


#%% BATCH PROCESS

# # PARAMETERS # #
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload' # your base directory
ROUND = True # Toggle if want to round force plate data to no decimals for checking
# # # # # # # # #


# Get subject IDs.
datadir = os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices')
subjs = os.listdir(datadir)
subjs.remove('.DS_Store')

#subjs = ['102913_315']

# Loop through subjects.
for subj in subjs:
    
    # Get list of all subject's files.
    subjdir = os.path.join(BASE_DIR, 'data', 'raw', 'ACL_DownSampledFiles', subj)
    subj_allfiles = os.listdir(subjdir)
    
    # Get list of trial names (number of trials and tasks is different for some subjects).
    downsampled_files = [filename for filename in subj_allfiles if 'Fs60' in filename]
    split_filenames = [substring.split('grf') for substring in downsampled_files if 'grf' in substring] # each GRF file has a corresponding IK file
    trialnames = [trialname[0][:-1][8:] for trialname in split_filenames] # Remove "Trimmed_" and trailing "_"
    
    for trial in trialnames:
        # Load in GR data.
        grpath = os.path.join(BASE_DIR, 'data', 'raw', 'ACL_DownSampledFiles', subj, ('Trimmed_' + trial + '_grf_Fs60.mot'))
        grpath = os.path.normcase(grpath)
        _, labels, data = readMotionFile(grpath)
        data = np.asarray(data)
        num_timesteps = data.shape[0]
        
        # Get GR data:
        grf_1_ix = [labels.index('1_ground_force_vx'), labels.index('1_ground_force_vy'), labels.index('1_ground_force_vz')]
        grf_2_ix = [labels.index('2_ground_force_vx'), labels.index('2_ground_force_vy'), labels.index('2_ground_force_vz')]

        # Put corresponding columns into each output's matrix.
        grf_1 = data[:,grf_1_ix]
        grf_2 = data[:,grf_2_ix]
        
        # Check force plates:
        grf_1_y = grf_1[:,1]
        grf_2_y = grf_2[:,1]
        
        grf_1_y = np.where(grf_1_y < 1, 0, grf_1_y) # zero out small signals
        grf_2_y = np.where(grf_2_y < 1, 0, grf_2_y)
        #grf_1_y = np.around(grf_1_y, decimals=0) # round to nearest whole number
        #grf_2_y = np.around(grf_2_y, decimals=0)
        
        if '_dj' in grpath:
            if np.count_nonzero(grf_1_y) < int(num_timesteps * 0.25) or np.count_nonzero(grf_2_y) < int(num_timesteps * 0.25):
                print(f"Error for Subject {subj}, Trial {trial}:")
                print(f"Num nonzero grf_1_y: {np.count_nonzero(grf_1_y)}")
                print(f"Num nonzero grf_2_y: {np.count_nonzero(grf_2_y)}")
                print(f"Num timesteps: {num_timesteps}")
        else: # exactly one leg should have no signal for single leg tasks
            #if (np.count_nonzero(grf_1_y) < int(num_timesteps * 0.25) and np.count_nonzero(grf_2_y) < int(num_timesteps * 0.25)) or (np.count_nonzero(grf_1_y) > int(num_timesteps * 0.25) and np.count_nonzero(grf_2_y) > int(num_timesteps * 0.25)):
            if np.count_nonzero(grf_2_y) < int(num_timesteps * 0.25):
                print(f"Error for Subject {subj}, Trial {trial}:")
                print(f"Num nonzero grf_1_y: {np.count_nonzero(grf_1_y)}")
                print(f"Num nonzero grf_2_y: {np.count_nonzero(grf_2_y)}")
                print(f"Num timesteps: {num_timesteps}")
                