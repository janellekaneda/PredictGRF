# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 23:02:54 2022

@author: jkaneda

Normalize the main model data structures and output new data structures.

EDIT: Sun Jan 8 2023: edit indices, etc. to work for new dataset shapes for 
out of distribution experiment. --> 
EDIT: Mon Jun 19 2023: for dowmsampled 100 Hz datasets with other changes
EDIT: Thu Jul 06 2023: subsample previous Y datasets for new interoutput structure, and normalize accordingly
"""

#%% IMPORTS
import os
import numpy as np
from normalize_dataset_utils import *

#%% 

# # PARAMETERS # #
MASK_VALUE = 999
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload' # your base directory
# # # # # # # # # 

# Specify experiment type (with or without physics inputs).
exp_type = 'with_physicsinputs' # technically should be the same means but just removing some of the means

# Load all built data.
X_train = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', exp_type, 'X_train_100Hz_RawDist.npy'), allow_pickle=True)
Y_train = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', exp_type, 'Y_train_100Hz_RawDist.npy'), allow_pickle=True)
X_train_cutting = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', exp_type, 'X_train_cutting_100Hz_RawDist.npy'), allow_pickle=True)
Y_train_cutting = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', exp_type, 'Y_train_cutting_100Hz_RawDist.npy'), allow_pickle=True)
X_dev = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', exp_type, 'X_dev_100Hz_RawDist.npy'), allow_pickle=True)
Y_dev = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', exp_type, 'Y_dev_100Hz_RawDist.npy'), allow_pickle=True)
X_test = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', exp_type, 'X_test_100Hz_RawDist.npy'), allow_pickle=True)
Y_test = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', exp_type, 'Y_test_100Hz_RawDist.npy'), allow_pickle=True)

# Reformulate the Y data structs:
Y_data_structs = [Y_train, Y_train_cutting, Y_dev, Y_test] 
Y_data_structs_new = []
for Y_data in Y_data_structs:

    Y_total_grf = np.expand_dims(Y_data[:,:,16], axis=-1) * Y_data[:,:,17:20]
    Y_total_grf[Y_total_grf == MASK_VALUE**2] = MASK_VALUE # reset mask value

    Y_class_r = np.expand_dims(Y_data[:,:,20], axis=-1)
    Y_class_l = np.expand_dims(Y_data[:,:,21], axis=-1)

    Y_dist_r = Y_data[:,:,22:25]

    Y_grf_r = np.expand_dims(Y_data[:,:,0], axis=-1) * Y_data[:,:,1:4]
    Y_grf_r[Y_grf_r == MASK_VALUE**2] = MASK_VALUE # reset mask value
    Y_grf_l = np.expand_dims(Y_data[:,:,4], axis=-1) * Y_data[:,:,5:8]
    Y_grf_l[Y_grf_l == MASK_VALUE**2] = MASK_VALUE # reset mask value

    # Put back into one struct
    Y_data_structs_new.append(np.concatenate((Y_total_grf, Y_class_r, Y_class_l, Y_dist_r, Y_grf_r, Y_grf_l), axis=-1))

Y_train = Y_data_structs_new[0]
Y_train_cutting = Y_data_structs_new[1]
Y_dev = Y_data_structs_new[2]
Y_test = Y_data_structs_new[3]

# check shape
print(f"Y_train shape: {Y_train.shape}")

# Get the training means and stds.
X_train_means, Y_train_means, X_train_stds, Y_train_stds = calculate_train_mean_std(X_train, Y_train, MASK_VALUE)
X_train_cutting_means, Y_train_cutting_means, X_train_cutting_stds, Y_train_cutting_stds = calculate_train_mean_std(X_train_cutting, Y_train_cutting, MASK_VALUE)

# Normalize each dataset.
X_train_norm, Y_train_norm = normalize_dataset(X_train, Y_train, X_train_means, Y_train_means, X_train_stds, Y_train_stds, MASK_VALUE)
X_dev_norm, Y_dev_norm = normalize_dataset(X_dev, Y_dev, X_train_means, Y_train_means, X_train_stds, Y_train_stds, MASK_VALUE)
X_test_norm, Y_test_norm = normalize_dataset(X_test, Y_test, X_train_means, Y_train_means, X_train_stds, Y_train_stds, MASK_VALUE)

X_train_cutting_norm, Y_train_cutting_norm = normalize_dataset(X_train_cutting, Y_train_cutting, X_train_cutting_means, Y_train_cutting_means, X_train_cutting_stds, Y_train_cutting_stds, MASK_VALUE)
X_dev_cutting_norm, Y_dev_cutting_norm = normalize_dataset(X_dev, Y_dev, X_train_cutting_means, Y_train_cutting_means, X_train_cutting_stds, Y_train_cutting_stds, MASK_VALUE)
X_test_cutting_norm, Y_test_cutting_norm = normalize_dataset(X_test, Y_test, X_train_cutting_means, Y_train_cutting_means, X_train_cutting_stds, Y_train_cutting_stds, MASK_VALUE)

# Re-put the binary step classification labels. Python ix are 3 and 5.
Y_train_norm[:,:,3:5] = Y_train[:,:,3:5]
Y_dev_norm[:,:,3:5] = Y_dev[:,:,3:5]
Y_test_norm[:,:,3:5] = Y_test[:,:,3:5]

Y_train_cutting_norm[:,:,3:5] = Y_train_cutting[:,:,3:5]
Y_dev_cutting_norm[:,:,3:5] = Y_dev[:,:,3:5]
Y_test_cutting_norm[:,:,3:5] = Y_test[:,:,3:5]

# Re-put the force distribution values since these are already 0 to 1.
Y_train_norm[:,:,5:8] = Y_train[:,:,5:8]
Y_dev_norm[:,:,5:8] = Y_dev[:,:,5:8]
Y_test_norm[:,:,5:8] = Y_test[:,:,5:8]

Y_train_cutting_norm[:,:,5:8] = Y_train_cutting[:,:,5:8]
Y_dev_cutting_norm[:,:,5:8] = Y_dev[:,:,5:8]
Y_test_cutting_norm[:,:,5:8] = Y_test[:,:,5:8]

# Specify output path.
outpath = os.path.join(BASE_DIR, 'data', 'processed', 'normalized', exp_type)

# Save normalized data structures.
#np.save(os.path.join(outpath, ('X_train_norm_100Hz_RawDist.npy')), X_train_norm)
np.save(os.path.join(outpath, ('Y_train_norm_100Hz_RawDist_InterOutputs.npy')), Y_train_norm)
#np.save(os.path.join(outpath, ('X_dev_norm_100Hz_RawDist.npy')), X_dev_norm)
np.save(os.path.join(outpath, ('Y_dev_norm_100Hz_RawDist_InterOutputs.npy')), Y_dev_norm)
#np.save(os.path.join(outpath, ('X_test_norm_100Hz_RawDist.npy')), X_test_norm)
np.save(os.path.join(outpath, ('Y_test_norm_100Hz_RawDist_InterOutputs.npy')), Y_test_norm)

#np.save(os.path.join(outpath, ('X_train_norm_100Hz_cutting_RawDist.npy')), X_train_cutting_norm)
np.save(os.path.join(outpath, ('Y_train_norm_100Hz_cutting_RawDist_InterOutputs.npy')), Y_train_cutting_norm)
#np.save(os.path.join(outpath, ('X_dev_norm_100Hz_cutting_RawDist.npy')), X_dev_cutting_norm)
np.save(os.path.join(outpath, ('Y_dev_norm_100Hz_cutting_RawDist_InterOutputs.npy')), Y_dev_cutting_norm)
#np.save(os.path.join(outpath, ('X_test_norm_100Hz_cutting_RawDist.npy')), X_test_cutting_norm)
np.save(os.path.join(outpath, ('Y_test_norm_100Hz_cutting_RawDist_InterOutputs.npy')), Y_test_cutting_norm)

# Save training means and stds.
#np.save(os.path.join(outpath, ('X_train_means_100Hz_RawDist.npy')), X_train_means)
np.save(os.path.join(outpath, ('Y_train_means_100Hz_RawDist_InterOutputs.npy')), Y_train_means)
#np.save(os.path.join(outpath, ('X_train_stds_100Hz_RawDist.npy')), X_train_stds)
np.save(os.path.join(outpath, ('Y_train_stds_100Hz_RawDist_InterOutputs.npy')), Y_train_stds)

#np.save(os.path.join(outpath, ('X_train_means_100Hz_cutting_RawDist.npy')), X_train_cutting_means)
np.save(os.path.join(outpath, ('Y_train_means_100Hz_cutting_RawDist_InterOutputs.npy')), Y_train_cutting_means)
#np.save(os.path.join(outpath, ('X_train_stds_100Hz_cutting_RawDist.npy')), X_train_cutting_stds)
np.save(os.path.join(outpath, ('Y_train_stds_100Hz_cutting_RawDist_InterOutputs.npy')), Y_train_cutting_stds)

# Verify sizes
print(f"X_train len: {X_train.shape[0]}")
print(f"X_dev len: {X_dev.shape[0]}")
print(f"X_test len: {X_test.shape[0]}")
