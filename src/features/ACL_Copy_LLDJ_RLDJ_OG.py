# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 22:03:55 2023

@author: jkaneda

Copy the original LLDJ and RLDJ feature matrix .npy files in ACL dataset.
Originals included Plate 1 data, but should be 0.
"""

#%% IMPORTS

import shutil
import os

#%% BATCH PROCESS

# # PARAMETERS # #
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload' # your base directory
# # # # # # # # #

datadir = os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices')
newdir = os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices_LLDJ_RLDJ_Error2_OG')

# Get subject IDs.
subjs = os.listdir(datadir)
subjs.remove('.DS_Store')

#subjs = ['102913_315']

# Loop through subjects.
for subj in subjs:
    
    # Get all LLDJ and RLDJ file names.
    feat_mats = os.listdir(os.path.join(datadir, subj))
    target_trials = [feat_mat for feat_mat in feat_mats if '_lldj' in feat_mat.lower() or '_rldj' in feat_mat.lower()]
    
    # Make a subject folder in newdir.
    os.makedirs(os.path.join(newdir, subj))
    
    # Copy over the files!
    for trial in target_trials:
        shutil.copy2(os.path.join(datadir, subj, trial),
                     os.path.join(newdir, subj, trial))


