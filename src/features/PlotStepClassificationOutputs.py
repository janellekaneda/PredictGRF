# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:22:10 2023

@author: jkaneda

To check odd contact/no contact trials in ACL dataset 
"""

#%% IMPORTS

import numpy as np
import os
import matplotlib.pyplot as plt

#%% PROCESS

TYPE = 'with_physicsinputs' # is same for both types but can check that
#TYPE = 'without_physicsinputs'
MAX_TIMESTEPS = 67
MASK_VALUE = 999
RESULTS_DIR = r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\results\step_classification\contact_ratios_plots"

# Load data
Y_dev = np.load(os.path.join(r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\processed\not_normalized",TYPE,"Y_dev_stepclass.npy"))
Y_test = np.load(os.path.join(r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\processed\not_normalized",TYPE,"Y_test_stepclass.npy"))
Y_train = np.load(os.path.join(r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\processed\not_normalized",TYPE,"Y_train_stepclass.npy"))

# Get number of trials per datasets
def get_num_subjects(ref_list):
    with open(ref_list, 'r') as file:
        lines = file.readlines()
        return len(lines)

# Plotting function
def plot(R_leg, L_leg, filename):
    plt.figure()
    plt.plot(R_leg, label='R')
    plt.plot(L_leg, label='L')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    
# Load lists of subjs per dataset
ref_list_paths = [r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\references\ACL_dev_subjs.txt",
                  r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\references\ACL_test_subjs.txt",
                  r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\references\ACL_train_subjs.txt"]
num_subjs = np.zeros(len(ref_list_paths), dtype=int) # follows order above
for ix, ref_list in enumerate(ref_list_paths):
    num_subjs[ix] = get_num_subjects(ref_list)

datasets = [Y_dev, Y_test, Y_train]
dataset_names = ['dev', 'test', 'train']
acl_tasks = ['dj','lldj','rldj']

for i,data in enumerate(datasets):
    if dataset_names[i] != 'train':
        num_tasks = 5
    else:
        num_tasks = 3
    # Just get ACL data
    data = data[-num_subjs[i]*num_tasks*3:,:,:]
    # Reshape
    data = np.reshape(data, (num_subjs[i], num_tasks, 3, MAX_TIMESTEPS, 2))
    # Only get the jump tasks for non-training
    if dataset_names[i] != 'train':
        data = data[:, 1:4, :, :, :]
    
    for subj in range(num_subjs[i]):
        for task in range(len(acl_tasks)):
            for trial in range(3):
                # Get unmasked
                R_leg = data[subj, task, trial, :, 0]
                L_leg = data[subj, task, trial, :, 1]
                
                if MASK_VALUE in R_leg:
                    min_timestep = np.min(np.nonzero(R_leg == MASK_VALUE))
                else:
                    min_timestep = MAX_TIMESTEPS
                
                R_leg = R_leg[0:min_timestep]
                L_leg = L_leg[0:min_timestep]
                    
                # Plot all test cases
                if acl_tasks[task] == 'dj':
                    if 0 in R_leg:
                        if 0 in L_leg:
                            plot(R_leg, L_leg, 'subj' + str(subj) + '_dj_trial' + str(trial) + '.png')
                
                if acl_tasks[task] == 'lldj':
                    if 0 in L_leg:
                        plot(R_leg, L_leg, 'subj' + str(subj) + '_lldj_trial' + str(trial) + '.png')
                        
                if acl_tasks[task] == 'rldj':
                    if 0 in R_leg:
                        plot(R_leg, L_leg, 'subj' + str(subj) + '_rldj_trial' + str(trial) + '.png')
