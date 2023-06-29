# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 21:28:58 2023

@author: jkaneda

Check created datasets for min/max values, etc.
"""
import os
import numpy as np

basedir = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\processed'
MASK_VALUE = 999

#norm_types = ['not_normalized']
norm_types = ['normalized']
phys_types = ['with_physicsinputs', 'without_physicsinputs']
# data_names = ['X_train_100Hz.npy','Y_train_100Hz.npy',
#               'X_train_cutting_100Hz.npy','Y_train_cutting_100Hz.npy',
#               'X_dev_100Hz.npy','Y_dev_100Hz.npy',
#               'X_test_100Hz.npy','Y_test_100Hz.npy']
data_names = ['X_train_norm_100Hz.npy','Y_train_norm_100Hz.npy',
              'X_train_norm_100Hz_cutting.npy','Y_train_norm_100Hz_cutting.npy',
              'X_dev_norm_100Hz.npy','Y_dev_norm_100Hz.npy',
              'X_dev_norm_100Hz_cutting.npy','Y_dev_norm_100Hz_cutting.npy',
              'X_test_norm_100Hz.npy','Y_test_norm_100Hz.npy',
              'X_test_norm_100Hz_cutting.npy','Y_test_norm_100Hz_cutting.npy']

for norm in norm_types:
    for phys in phys_types:
        for data_name in data_names:
            data = np.load(os.path.join(basedir, norm, phys, data_name))
            data_mask = (data != MASK_VALUE)
            data = data[data_mask]
            print(data_name)
            #print(data.shape)
            print(f"min: {np.min(data)}")
            print(f"max: {np.max(data)}")
            
            # if 'X' in data_name:
            #     print(f"max ix: {np.argmax(data)}")
        
        