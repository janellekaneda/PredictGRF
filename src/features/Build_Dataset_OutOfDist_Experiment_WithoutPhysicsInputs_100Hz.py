# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 02:01:15 2023

@author: jkaneda

Simply partition previously-generated datasets from "Build_Dataset...WithPhysicsInputs.py"
to remove whole-body COM inputs.
"""

#%% IMPORTS

import numpy as np
import os

#%%

# # PARAMETERS # #
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload' # your base directory
WITH_CUTTING = True
# # # # # # # # # 

# Load "X" datasets (only removing input features, not output features).
if WITH_CUTTING:
    X_train_phys = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'with_physicsinputs', 'X_train_cutting_100Hz_RawDist.npy'), allow_pickle=True)
else:
    X_train_phys = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'with_physicsinputs', 'X_train_100Hz_RawDist.npy'), allow_pickle=True)

X_dev_phys = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'with_physicsinputs', 'X_dev_100Hz_RawDist.npy'), allow_pickle=True)
X_test_phys = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'with_physicsinputs', 'X_test_100Hz_RawDist.npy'), allow_pickle=True)

#%% Remove COM features (feature numbers 13-20, if first feature index = 1 --> index 12-20 in Python)
target_feats = list(range(36))
[target_feats.remove(x) for x in list(range(12,20))] # remove COM features

X_train = X_train_phys[:,:,target_feats]
X_dev = X_dev_phys[:,:,target_feats]
X_test = X_test_phys[:,:,target_feats]

# Save!
outpath = os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'without_physicsinputs')
if WITH_CUTTING:
    np.save(os.path.join(outpath, 'X_train_cutting_100Hz_RawDist.npy'), X_train)
else:
    np.save(os.path.join(outpath, 'X_train_100Hz_RawDist.npy'), X_train)
np.save(os.path.join(outpath, 'X_dev_100Hz_RawDist.npy'), X_dev)
np.save(os.path.join(outpath, 'X_test_100Hz_RawDist.npy'), X_test)

# *** (Manually copy over "Y" datasets from with_physicsinputs to without_physicsinputs) ***