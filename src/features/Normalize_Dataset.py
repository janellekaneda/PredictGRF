# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 23:02:54 2022

@author: jkaneda

Normalize the main model data structures and output new data structures.
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

# Load all built data.
X_train = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'X_train.npy'), allow_pickle=True)
Y_train = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'Y_train.npy'), allow_pickle=True)
X_dev = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'X_dev.npy'), allow_pickle=True)
Y_dev = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'Y_dev.npy'), allow_pickle=True)
X_test = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'X_test.npy'), allow_pickle=True)
Y_test = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'Y_test.npy'), allow_pickle=True)

# Subset data to create separate datasets for OAGR and ACL, normalized to their respective training means.
#data_type = 'oagr'
data_type = 'acl'
if data_type == 'oagr':
    X_train = X_train[0:5400,:,:]
    Y_train = Y_train[0:5400,:,:]
    X_dev = X_dev[0:700,:,:]
    Y_dev = Y_dev[0:700,:,:]
    X_test = X_test[0:700,:,:]
    Y_test = Y_test[0:700,:,:]
else:
    X_train = X_train[5400:,:,:]
    Y_train = Y_train[5400:,:,:]
    X_dev = X_dev[700:,:,:]
    Y_dev = Y_dev[700:,:,:]
    X_test = X_test[700:,:,:]
    Y_test = Y_test[700:,:,:]
    

# Get the training means and stds.
X_train_means, Y_train_means, X_train_stds, Y_train_stds = calculate_train_mean_std(X_train, Y_train, MASK_VALUE)

# Normalize each dataset.
X_train_norm, Y_train_norm = normalize_dataset(X_train, Y_train, X_train_means, Y_train_means, X_train_stds, Y_train_stds, MASK_VALUE)
X_dev_norm, Y_dev_norm = normalize_dataset(X_dev, Y_dev, X_train_means, Y_train_means, X_train_stds, Y_train_stds, MASK_VALUE)
X_test_norm, Y_test_norm = normalize_dataset(X_test, Y_test, X_train_means, Y_train_means, X_train_stds, Y_train_stds, MASK_VALUE)

# Specify output path.
outpath = os.path.join(BASE_DIR, 'data', 'processed', 'normalized')

# Save normalized data structures.
np.save(os.path.join(outpath, ('X_train_norm_' + data_type + '.npy')), X_train_norm)
np.save(os.path.join(outpath, ('Y_train_norm_' + data_type + '.npy')), Y_train_norm)
np.save(os.path.join(outpath, ('X_dev_norm_' + data_type + '.npy')), X_dev_norm)
np.save(os.path.join(outpath, ('Y_dev_norm_' + data_type + '.npy')), Y_dev_norm)
np.save(os.path.join(outpath, ('X_test_norm_' + data_type + '.npy')), X_test_norm)
np.save(os.path.join(outpath, ('Y_test_norm_' + data_type + '.npy')), Y_test_norm)

# Save training means and stds.
np.save(os.path.join(outpath, ('X_train_means_' + data_type + '.npy')), X_train_means)
np.save(os.path.join(outpath, ('Y_train_means_' + data_type + '.npy')), Y_train_means)
np.save(os.path.join(outpath, ('X_train_stds_' + data_type + '.npy')), X_train_stds)
np.save(os.path.join(outpath, ('Y_train_stds_' + data_type + '.npy')), Y_train_stds)

# Verify sizes
print(f"X_train len: {X_train.shape[0]}")
print(f"X_dev len: {X_dev.shape[0]}")
print(f"X_test len: {X_test.shape[0]}")

