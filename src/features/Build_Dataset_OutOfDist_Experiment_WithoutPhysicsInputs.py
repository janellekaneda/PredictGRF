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
# # # # # # # # # 

# Load "X" datasets (only removing input features, not output features).
X_train_phys = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'with_physicsinputs', 'X_train.npy'), allow_pickle=True)
X_dev_phys = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'with_physicsinputs', 'X_dev.npy'), allow_pickle=True)
X_test_phys = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'with_physicsinputs', 'X_test.npy'), allow_pickle=True)

#%% Remove COM features (feature numbers 19-26, if first feature index = 1 --> index 18-25 in Python)
target_feats = list(range(42))
[target_feats.remove(x) for x in list(range(18,26))] # remove COM features

X_train = X_train_phys[:,:,target_feats]
X_dev = X_dev_phys[:,:,target_feats]
X_test = X_test_phys[:,:,target_feats]

# Save!
outpath = os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'without_physicsinputs')
np.save(os.path.join(outpath, 'X_train.npy'), X_train)
np.save(os.path.join(outpath, 'X_dev.npy'), X_dev)
np.save(os.path.join(outpath, 'X_test.npy'), X_test)

# (Manually copy over "Y" datasets)


