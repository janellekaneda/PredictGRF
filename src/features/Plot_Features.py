# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:07:44 2023

@author: jkaneda

Visualize input/output features in the given dataset and trial number.
"""

#%% IMPORTS
import os
import numpy as np

#%% 

# # PARAMETERS # #
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload' # your base directory
# # # # # # # # # 

exp_type = 'with_physicsinputs'
trial_num = 0


datadir = os.path.join(BASE_DIR, 'data', 'processed', 'normalized', exp_type)
dataset = np.load(os.path.join(datadir, 'X_train_norm.npy'), allow_pickle=True)

data_to_plot = 
