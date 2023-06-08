# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 23:50:32 2023

@author: jkaneda

This script loads in the previously built datasets and creates step binary
contact/no contact output features for step classification.
"""
#%% IMPORTS

import numpy as np
import os

#%% PROCESS

#TYPE = 'with_physicsinputs' # toggle
TYPE = 'without_physicsinputs'
MAX_TIMESTEPS = 67
MASK_VALUE = 999

# load (unnormalized) data to classify contact
Y_dev_og = np.load(os.path.join(r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\processed\not_normalized",TYPE,"Y_dev.npy"))
Y_test_og = np.load(os.path.join(r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\processed\not_normalized",TYPE,"Y_test.npy"))
Y_train_og = np.load(os.path.join(r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\processed\not_normalized",TYPE,"Y_train.npy"))

# Loop through each set
datasets = [Y_dev_og, Y_test_og, Y_train_og]
outnames = ['dev', 'test', 'train']

for i, data in enumerate(datasets):
    num_trials = data.shape[0]
    # Init output arr
    output = np.ones(shape=(num_trials,MAX_TIMESTEPS, 2)) * MASK_VALUE # right leg, then left leg for feature order 
    for trial in range(num_trials):
        # Get the R_mag and L_mag for the GRFs
        R_mag = data[trial,:,0]
        L_mag = data[trial,:,4]
        
        # Get the non-masked values
        if MASK_VALUE in R_mag:
            min_timestep = np.min(np.nonzero(R_mag == MASK_VALUE))
        else:
            min_timestep = MAX_TIMESTEPS
        R_mag = R_mag[0:min_timestep]
        L_mag = L_mag[0:min_timestep]
        
        # Boolean mask, where 1 = contact, 0 = no contact, and store
        output[trial, 0:min_timestep, 0] = np.where(R_mag > 0, 1, 0)
        output[trial, 0:min_timestep, 1] = np.where(L_mag > 0, 1, 0)
    
    # Save the classification outputs for each dataset
    np.save(os.path.join(r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\processed\not_normalized",TYPE,"Y_" + outnames[i] + "_stepclass.npy"), output)
            