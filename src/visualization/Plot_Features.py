# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:07:44 2023

@author: jkaneda

Visualize input/output features in the given dataset and trial number.
"""

#%% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt

#%% 

# # PARAMETERS # #
MASK_VALUE = 999
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload' # your base directory
# # # # # # # # # 

exp_type = 'with_physicsinputs'
trial_num = 817
dataset_name = 'Y_dev_norm.npy'

# Load data
datadir = os.path.join(BASE_DIR, 'data', 'processed', 'normalized', exp_type)
dataset = np.load(os.path.join(datadir, dataset_name), allow_pickle=True)

# Identify the minimum where masking/padding started for the given subject over all trials for given task.
min_timestep = np.min(np.nonzero(dataset == MASK_VALUE)[1])

data_to_plot = dataset[trial_num, 0:min_timestep, :]

if 'X' in dataset_name: # plotting input features
    plt.figure()
    plt.plot(data_to_plot[:,0:18], 'b') # IK features
    plt.plot(data_to_plot[:,18:26], 'r') # COM features
    plt.plot(data_to_plot[:,26:42], 'g') # foot features

if 'Y' in dataset_name: # plotting output features
    plt.figure()
    plt.plot(data_to_plot[:,0:4], 'r') # right GRF
    plt.plot(data_to_plot[:,4:8], 'b') # left GRF
    plt.plot(data_to_plot[:,8:12], 'r--') # right GRM
    plt.plot(data_to_plot[:,12:16], 'b--') # left GRM
