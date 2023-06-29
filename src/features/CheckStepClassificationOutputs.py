# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:35:09 2023

@author: jkaneda

This script checks the percentages of contact/no contact in each dataset
"""

#%% IMPORTS

import numpy as np
import os

#%% PROCESS

TYPE = 'with_physicsinputs' # is same for both types but can check that
#TYPE = 'without_physicsinputs'
MAX_TIMESTEPS = 67
MASK_VALUE = 999

# Load data
Y_dev = np.load(os.path.join(r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\processed\not_normalized",TYPE,"Y_dev_stepclass.npy"))
Y_test = np.load(os.path.join(r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\processed\not_normalized",TYPE,"Y_test_stepclass.npy"))
Y_train = np.load(os.path.join(r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\processed\not_normalized",TYPE,"Y_train_stepclass.npy"))

# Get number of trials per datasets
def get_num_subjects(ref_list):
    with open(ref_list, 'r') as file:
        lines = file.readlines()
        return len(lines)
    
# Load lists of subjs per dataset
ref_list_paths = [r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\references\OAGR_dev_subjs.txt",
                  r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\references\OAGR_test_subjs.txt",
                  r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\references\OAGR_train_subjs.txt",
                  r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\references\ACL_dev_subjs.txt",
                  r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\references\ACL_test_subjs.txt",
                  r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\references\ACL_train_subjs.txt"]
num_subjs = np.zeros(len(ref_list_paths), dtype=int) # follows order above
for ix, ref_list in enumerate(ref_list_paths):
    num_subjs[ix] = get_num_subjects(ref_list)

# # Calc ratios # #
datasets = [Y_dev, Y_test, Y_train]
dataset_names = ['dev', 'test', 'train']
acl_tasks = ['cutting','dj','lldj','rldj','unant_cut']
acl_tasks_train = ['dj','lldj','rldj']

print(num_subjs)

for i, data in enumerate(datasets):
    # # Overall # #
    unique, counts = np.unique(data, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    print(f"Ratio of contact:no contact for total {dataset_names[i]} is {np.round(counts_dict[1]/counts_dict[0],2)}")
    
    # # Per leg # #
    R_data = data[:,:,0]
    L_data = data[:,:,1]
    unique, counts = np.unique(R_data, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    print(f"Ratio of contact:no contact for right leg {dataset_names[i]} is {np.round(counts_dict[1]/counts_dict[0],2)}")
    unique, counts = np.unique(L_data, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    print(f"Ratio of contact:no contact for left leg {dataset_names[i]} is {np.round(counts_dict[1]/counts_dict[0],2)}")
    
    # # Per leg per task # #
    R_data_walking = R_data[0:num_subjs[i]*100,:]
    L_data_walking = L_data[0:num_subjs[i]*100,:]
    R_data_acl     = R_data[num_subjs[i]*100:,:]
    L_data_acl     = L_data[num_subjs[i]*100:,:]
    
    # Walking
    unique, counts = np.unique(R_data_walking, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    print(f"Ratio of contact:no contact for right leg walking {dataset_names[i]} is {np.round(counts_dict[1]/counts_dict[0],2)}")
    unique, counts = np.unique(L_data_walking, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    print(f"Ratio of contact:no contact for left leg walking {dataset_names[i]} is {np.round(counts_dict[1]/counts_dict[0],2)}")
    
    # ACL tasks

    if dataset_names[i] != 'train':
        num_tasks = 5
        task_names = acl_tasks
    else:
        num_tasks = 3
        task_names = acl_tasks_train
        
    R_data_acl = np.reshape(R_data_acl, (num_subjs[i+3], num_tasks, 3, R_data_acl.shape[-1])) # subjs x tasks x trials x timesteps
    L_data_acl = np.reshape(L_data_acl, (num_subjs[i+3], num_tasks, 3, L_data_acl.shape[-1]))
        
    for task in range(num_tasks):
       # Right leg
        unique, counts = np.unique(R_data_acl[:,task,:,:], return_counts=True)
        counts_dict = dict(zip(unique, counts))
        if 1 in counts_dict and 0 in counts_dict:
            print(f"Ratio of contact:no contact for right leg {task_names[task]} {dataset_names[i]} is {np.round(counts_dict[1]/counts_dict[0],2)}")
            print(f"no. of timesteps with contact for right leg {task_names[task]} {dataset_names[i]}: {counts_dict[1]}")
            print(f"no. of timesteps without contact: {counts_dict[0]}")
        elif 1 in counts_dict:
            print(f"Ratio of contact:no contact for right leg {task_names[task]} {dataset_names[i]} is all contact")
        elif 0 in counts_dict:
            print(f"Ratio of contact:no contact for right leg {task_names[task]} {dataset_names[i]} is all no contact")
        # Left leg
        unique, counts = np.unique(L_data_acl[:,task,:,:], return_counts=True)
        counts_dict = dict(zip(unique, counts))
        if 1 in counts_dict and 0 in counts_dict:
            print(f"Ratio of contact:no contact for left leg {task_names[task]} {dataset_names[i]} is {np.round(counts_dict[1]/counts_dict[0],2)}")
            print(f"no. of timesteps with contact for left leg {task_names[task]} {dataset_names[i]}: {counts_dict[1]}")
            print(f"no. of timesteps without contact: {counts_dict[0]}")
        elif 1 in counts_dict:
            print(f"Ratio of contact:no contact for left leg {task_names[task]} {dataset_names[i]} is all contact")
        elif 0 in counts_dict:
            print(f"Ratio of contact:no contact for left leg {task_names[task]} {dataset_names[i]} is all no contact")
        
            
    
    

    