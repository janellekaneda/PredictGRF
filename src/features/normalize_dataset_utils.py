# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 22:53:18 2022

This module contains functions to calculate the training mean and standard
deviation, as well as using these calculated values to normalize the train,
dev, and test datasets.

@author: jkaneda
"""

#%% IMPORTS

import numpy as np

#%% Calculate training mean and standard devation.

def calculate_train_mean_std(X_train, Y_train, MASK_VALUE):
    """
    Calculates the training set mean and standard deviation for each input
    and output feature.
    
    Inputs:
        X_train: (numpy array) un-normalized training features (size: num training ex, max time steps, num features)
        Y_train: (numpy array) un-normalized training outputs (size; num trainigng ex, max time steps, num outputs)
        MASK_VALUE: (int) number used as the masking value (e.g. 999 or 123)
        
    Returns:
        X_train_means: (numpy array) array of each mean for each input feature (size: 1 x num features)
        Y_train_means: (numpy array) array of each mean for each output feature (size: 1 x num outputs)
        X_train_stds: (numpy array) array of each standard deviation for each input feature (size: 1 x num features)
        Y_train_stds: (numpy array) array of each standard deviation for each output feature (size: 1 x num outputs)
    """
    # # PROCESS INPUT STRUCTURES # # 
    
    # Make a mask of all non-mask value items (True = real data, False = mask data).
    X_train_mask = (X_train != MASK_VALUE)
    Y_train_mask = (Y_train != MASK_VALUE)
    # Find the mean.
    X_train_means = np.mean(X_train[X_train_mask].reshape(-1,X_train.shape[-1]), axis=0) # shape: (num trials x num timesteps per trial) x num feats
    Y_train_means = np.mean(Y_train[Y_train_mask].reshape(-1,Y_train.shape[-1]), axis=0)
    # Find the std.
    X_train_stds = np.std(X_train[X_train_mask].reshape(-1,X_train.shape[-1]), axis=0)
    Y_train_stds = np.std(Y_train[Y_train_mask].reshape(-1,Y_train.shape[-1]), axis=0)
    
    # Reshape the arrays.
    X_train_means = X_train_means.reshape(len(X_train_means), 1).T
    Y_train_means = Y_train_means.reshape(len(Y_train_means), 1).T
    X_train_stds = X_train_stds.reshape(len(X_train_stds), 1).T
    Y_train_stds = Y_train_stds.reshape(len(Y_train_stds), 1).T
    
    return X_train_means, Y_train_means, X_train_stds, Y_train_stds

#%% Normalize each dataset.

def normalize_dataset(X_dataset, Y_dataset, X_train_means, Y_train_means, X_train_stds, Y_train_stds, MASK_VALUE):
    """
    Normalize the X and Y datasets.
    
    Inputs:
        X_dataset: (numpy array) one of X_train, X_dev, or X_test
        Y_dataset: (numpy array) one of Y_train, Y_dev, or Y_test
        X_train_means: (numpy array) array of each mean for each input feature (size: 1 x num features)
        Y_train_means: (numpy array) array of each mean for each output feature (size: 1 x num outputs)
        X_train_stds: (numpy array) array of each standard deviation for each input feature (size: 1 x num features)
        Y_train_stds: (numpy array) array of each standard deviation for each output feature (size: 1 x num outputs)
        MASK_VALUE: (int) number used as the masking value (e.g. 999 or 123)
        
   Returns:
       X_dataset_norm: (numpy array) X_dataset normalized, same original shape
       Y_dataset_norm: (numpy array) Y_dataset normalized, same original shape
   """
    # Create a mask of all MASK value items (True = real data, False = mask data).
    X_dataset_mask = (X_dataset == MASK_VALUE)
    Y_dataset_mask = (Y_dataset == MASK_VALUE)
    
    # Calculate the normalized datasets (this will also normalize the masksed values, but we will reset these after).
    X_dataset_norm = (X_dataset - X_train_means) / X_train_stds
    Y_dataset_norm = (Y_dataset - Y_train_means) / Y_train_stds
    
    # Make any NaNs = 0 caused by 0 division.
    X_dataset_norm = np.nan_to_num(X_dataset_norm)
    Y_dataset_norm = np.nan_to_num(Y_dataset_norm)
   
    # Reset the masked values.
    X_dataset_norm[X_dataset_mask] = MASK_VALUE
    Y_dataset_norm[Y_dataset_mask] = MASK_VALUE
   
    return X_dataset_norm, Y_dataset_norm