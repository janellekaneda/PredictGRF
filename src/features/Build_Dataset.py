# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:26:21 2022

@author: jkaneda
"""

#%% IMPORTS

import os
import random
import numpy as np
from build_dataset_utils import *

#%% BATCH PROCESS

# # PARAMETERS # #
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload' # your base directory
MAX_TIMESTEPS = 168
MASK_VALUE = 999
NUM_INPUT_FEATS = 42
NUM_OUTPUT_FEATS = 16
NUM_OAGR_TRIALS = 100 # 20 steps x 5 conditions per subject
NUM_ACL_TRIALS = 15 # 3 trials x 5 tasks per subject, after upsampling
SEED = 43
# # # # # # # # # 

# Declare random seed.
random.seed(SEED)

# Condition / task names in each dataset for looping.
oagr_conds = ['baseline_TM1','eval_5deg1','eval_10deg1','eval_neg5deg1','eval_neg10deg1']
acl_tasks = ['_cutting','_dj','_lldj','_rldj','_unant_cut']

# # Get lists of train, dev, and test subject IDs for each dataset. # #
subjects_oagr = os.listdir(os.path.join(BASE_DIR, 'data', 'interim', 'OAGR_FeatureMatrices'))
#train_subjs_oagr, dev_subjs_oagr, test_subjs_oagr = get_train_dev_test_split(0.8, subjects_oagr, SEED)

# Instead of new split, use same train:dev:test split from milestone report to better compare previous and new model:
train_subjs_oagr_file = open(os.path.join(BASE_DIR, 'references', 'OAGR_train_subjs.txt'), 'r')
dev_subjs_oagr_file   = open(os.path.join(BASE_DIR, 'references', 'OAGR_dev_subjs.txt'), 'r')
test_subjs_oagr_file  = open(os.path.join(BASE_DIR, 'references', 'OAGR_test_subjs.txt'), 'r')

train_subjs_oagr_list = train_subjs_oagr_file.read().split('\n')
dev_subjs_oagr_list   = dev_subjs_oagr_file.read().split('\n')
test_subjs_oagr_list  = test_subjs_oagr_file.read().split('\n')
train_subjs_oagr_list.remove('') # remove trailing space
dev_subjs_oagr_list.remove('')
test_subjs_oagr_list.remove('')

train_subjs_oagr = [('Subject_' + id) for id in train_subjs_oagr_list]
dev_subjs_oagr = [('Subject_' + id) for id in dev_subjs_oagr_list]
test_subjs_oagr = [('Subject_' + id) for id in test_subjs_oagr_list]

# Get ACL subjects list.
subjects_acl =  os.listdir(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices'))
train_subjs_acl, dev_subjs_acl, test_subjs_acl = get_train_dev_test_split(0.8, subjects_acl, SEED)

#%%
# Write the lists of ACL split subjects to file.
with open(os.path.join(BASE_DIR, 'references', 'ACL_train_subjs.txt'), 'w') as f:
    for train_subj in train_subjs_acl:
        f.write(f"{train_subj}\n")
with open(os.path.join(BASE_DIR, 'references', 'ACL_dev_subjs.txt'), 'w') as f:
    for dev_subj in dev_subjs_acl:
        f.write(f"{dev_subj}\n")
with open(os.path.join(BASE_DIR, 'references', 'ACL_test_subjs.txt'), 'w') as f:
    for test_subj in test_subjs_acl:
        f.write(f"{test_subj}\n")

# # Initialize train, dev, and test arrays with mask value, and dataset identifier lists. # #
num_train_ex_oagr = len(train_subjs_oagr) * NUM_OAGR_TRIALS
num_train_ex_acl = len(train_subjs_acl) * NUM_ACL_TRIALS
X_train = np.ones((num_train_ex_oagr + num_train_ex_acl, MAX_TIMESTEPS, NUM_INPUT_FEATS)) * MASK_VALUE
Y_train = np.ones((num_train_ex_oagr + num_train_ex_acl, MAX_TIMESTEPS, NUM_OUTPUT_FEATS)) * MASK_VALUE
train_datalabels = np.concatenate([np.ones(num_train_ex_oagr), np.ones(num_train_ex_acl)*2]) # since append OAGR first and ACL second

num_dev_ex_oagr = len(dev_subjs_oagr) * NUM_OAGR_TRIALS
num_dev_ex_acl = len(dev_subjs_acl) * NUM_ACL_TRIALS
X_dev = np.ones((num_dev_ex_oagr + num_dev_ex_acl, MAX_TIMESTEPS, NUM_INPUT_FEATS)) * MASK_VALUE
Y_dev = np.ones((num_dev_ex_oagr + num_dev_ex_acl, MAX_TIMESTEPS, NUM_OUTPUT_FEATS)) * MASK_VALUE
dev_datalabels = np.concatenate([np.ones(num_dev_ex_oagr), np.ones(num_dev_ex_acl)*2])

num_test_ex_oagr = len(test_subjs_oagr) * NUM_OAGR_TRIALS
num_test_ex_acl = len(test_subjs_acl) * NUM_ACL_TRIALS
X_test = np.ones((num_test_ex_oagr + num_test_ex_acl, MAX_TIMESTEPS, NUM_INPUT_FEATS)) * MASK_VALUE
Y_test = np.ones((num_test_ex_oagr + num_test_ex_acl, MAX_TIMESTEPS, NUM_OUTPUT_FEATS)) * MASK_VALUE
test_datalabels = np.concatenate([np.ones(num_test_ex_oagr), np.ones(num_test_ex_acl)*2])

# # Loop over each split dataset, and append accordingly. # #
# Init ix to sequentially add data to each data array.
train_ix_all = 0
dev_ix_all = 0
test_ix_all = 0

# Loop over OAGR dataset first.
for subj in subjects_oagr:
    
    for cond in oagr_conds:
        
        # Get list of all feature matrix files for given subject and condition.
        cond_dir = os.path.join(BASE_DIR, 'data','interim', 'OAGR_FeatureMatrices', subj, cond)
        feats_list = os.listdir(cond_dir)
        
        for feats_filename in feats_list:
            
            # Load feature matrix.
            feats = np.load(os.path.join(BASE_DIR, 'data', 'interim', 'OAGR_FeatureMatrices', subj, cond, feats_filename), allow_pickle=True)
            
            # Get number of time steps for given trial.
            num_timesteps = feats.shape[0]
            
            # Extract X and Y (inputs and outputs).
            X_data = feats[:, 0:NUM_INPUT_FEATS] # shape = num timesteps x num input features
            Y_data = feats[:, NUM_INPUT_FEATS:(NUM_INPUT_FEATS+NUM_OUTPUT_FEATS)]
            
            # Append to each data split set based on what group the current subject is in.
            if subj in train_subjs_oagr:
                X_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0) # add axis for trial num
                Y_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                train_ix_all += 1 # Increment index.
            
            if subj in dev_subjs_oagr:
                X_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                Y_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                dev_ix_all += 1
            
            if subj in test_subjs_oagr:
                X_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                Y_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                test_ix_all += 1
    
    print(f"Finished {subj}")

# Next, loop over ACL dataset.
for subj in subjects_acl:
    
    # Get list of all feature matrix files for given subject.
    feats_list = os.listdir(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj))
    # Make feature mat list lowercase.
    feats_list_lc = [mat.lower() for mat in feats_list]
    
    if len(feats_list_lc) == 15: # append normally             
        
        for feats_filename in feats_list:
            
            # Load feature matrix.
            feats = np.load(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj, feats_filename), allow_pickle=True)
            
            # Get number of time steps for given trial.
            num_timesteps = feats.shape[0]
            
            # Extract X and Y (inputs and outputs).
            X_data = feats[:, 0:NUM_INPUT_FEATS] # shape = num timesteps x num input features
            Y_data = feats[:, NUM_INPUT_FEATS:(NUM_INPUT_FEATS+NUM_OUTPUT_FEATS)]
            
            # Append to each data split set based on what group the current subject is in.
            if subj in train_subjs_acl:
                X_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0) # add axis for trial num
                Y_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                train_ix_all += 1 # Increment index.
            
            if subj in dev_subjs_acl:
                X_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                Y_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                dev_ix_all += 1
            
            if subj in test_subjs_acl:
                X_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                Y_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                test_ix_all += 1
    
    else: # need to append additional trials with upsampling, retaining same task order
    
        for task in acl_tasks:
                
            task_count = sum(map(lambda x: task in x, feats_list_lc))
            task_filenames = [mat for mat in feats_list_lc if task in mat]
            
            if task_count == 3: # append normally
            
                for task_filename in task_filenames:
                    
                    # Load feature matrix.
                    feats = np.load(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj, task_filename), allow_pickle=True)
                    
                    # Get number of time steps for given trial.
                    num_timesteps = feats.shape[0]
                    
                    # Extract X and Y (inputs and outputs).
                    X_data = feats[:, 0:NUM_INPUT_FEATS] # shape = num timesteps x num input features
                    Y_data = feats[:, NUM_INPUT_FEATS:(NUM_INPUT_FEATS+NUM_OUTPUT_FEATS)]
                    
                    # Append to each data split set based on what group the current subject is in.
                    if subj in train_subjs_acl:
                        X_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0) # add axis for trial num
                        Y_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        train_ix_all += 1 # Increment index.
                    
                    if subj in dev_subjs_acl:
                        X_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                        Y_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        dev_ix_all += 1
                    
                    if subj in test_subjs_acl:
                        X_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                        Y_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        test_ix_all += 1
                    
            
            if task_count == 1: # repeat that task's single trial three times
                
                # Load feature matrix.
                feats = np.load(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj, task_filenames[0]), allow_pickle=True)
                
                # Get number of time steps for given trial.
                num_timesteps = feats.shape[0]
                
                # Extract X and Y (inputs and outputs).
                X_data = feats[:, 0:NUM_INPUT_FEATS] # shape = num timesteps x num input features
                Y_data = feats[:, NUM_INPUT_FEATS:(NUM_INPUT_FEATS+NUM_OUTPUT_FEATS)]
                
                # First, append that single trial.
                
                # Append to each data split set based on what group the current subject is in.
                if subj in train_subjs_acl:
                    X_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0) # add axis for trial num
                    Y_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                    train_ix_all += 1 # Increment index.
                
                if subj in dev_subjs_acl:
                    X_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                    Y_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                    dev_ix_all += 1
                
                if subj in test_subjs_acl:
                    X_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                    Y_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                    test_ix_all += 1
                
                # Then, upsample.
                for upsample in range(2):
                
                    # Append to each data split set based on what group the current subject is in.
                    if subj in train_subjs_acl:
                        X_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0) # add axis for trial num
                        Y_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        train_ix_all += 1 # Increment index.
                    
                    if subj in dev_subjs_acl:
                        X_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                        Y_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        dev_ix_all += 1
                    
                    if subj in test_subjs_acl:
                        X_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                        Y_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        test_ix_all += 1
            
            
            if task_count == 2: # randomly select one of the existing two trials, and upsample that
                
                # First, append the two trials.
                for task_filename in task_filenames:
                    
                    # Load feature matrix.
                    feats = np.load(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj, task_filename), allow_pickle=True)
                    
                    # Get number of time steps for given trial.
                    num_timesteps = feats.shape[0]
                    
                    # Extract X and Y (inputs and outputs).
                    X_data = feats[:, 0:NUM_INPUT_FEATS] # shape = num timesteps x num input features
                    Y_data = feats[:, NUM_INPUT_FEATS:(NUM_INPUT_FEATS+NUM_OUTPUT_FEATS)]
                    
                    # Append to each data split set based on what group the current subject is in.
                    if subj in train_subjs_acl:
                        X_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0) # add axis for trial num
                        Y_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        train_ix_all += 1 # Increment index.
                    
                    if subj in dev_subjs_acl:
                        X_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                        Y_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        dev_ix_all += 1
                    
                    if subj in test_subjs_acl:
                        X_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                        Y_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        test_ix_all += 1
            
                # Then, randomly select which one of the task trials to repeat.
                task_to_repeat_ix = random.randint(0,1)
                
                # Load feature matrix.
                feats = np.load(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj, task_filenames[task_to_repeat_ix]), allow_pickle=True)
                
                # Get number of time steps for given trial.
                num_timesteps = feats.shape[0]
                
                # Extract X and Y (inputs and outputs).
                X_data = feats[:, 0:NUM_INPUT_FEATS] # shape = num timesteps x num input features
                Y_data = feats[:, NUM_INPUT_FEATS:(NUM_INPUT_FEATS+NUM_OUTPUT_FEATS)]
                
                # Append to each data split set based on what group the current subject is in.
                if subj in train_subjs_acl:
                    X_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0) # add axis for trial num
                    Y_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                    train_ix_all += 1 # Increment index.
                
                if subj in dev_subjs_acl:
                    X_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                    Y_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                    dev_ix_all += 1
                
                if subj in test_subjs_acl:
                    X_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                    Y_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                    test_ix_all += 1
                    
    print(f"Finished {subj}")

# Save the train, dev, and test set arrays.
outpath = os.path.join(BASE_DIR, 'data', 'processed','not_normalized')
np.save(os.path.join(outpath, 'X_train.npy'), X_train)
np.save(os.path.join(outpath, 'Y_train.npy'), Y_train)
np.save(os.path.join(outpath, 'X_dev.npy'), X_dev)
np.save(os.path.join(outpath, 'Y_dev.npy'), Y_dev)
np.save(os.path.join(outpath, 'X_test.npy'), X_test)
np.save(os.path.join(outpath, 'Y_test.npy'), Y_test)

# Save lists of labels identifying which split dataset example belongs to each dataset (i.e., 'oagr' (1) or 'acl' (2) ).
np.save(os.path.join(BASE_DIR, 'data', 'processed', 'train_datalabels.npy'), train_datalabels)
np.save(os.path.join(BASE_DIR, 'data', 'processed', 'dev_datalabels.npy'), dev_datalabels)
np.save(os.path.join(BASE_DIR, 'data', 'processed', 'test_datalabels.npy'), test_datalabels)

# Check that appended correctly...
assert train_ix_all == num_train_ex_oagr + num_train_ex_acl, "ERROR: train indices don't match total number of training examples"
assert dev_ix_all == num_dev_ex_oagr + num_dev_ex_acl, "ERROR: dev indices don't match total number of training examples"
assert test_ix_all == num_test_ex_oagr + num_test_ex_acl, "ERROR: test indices don't match total number of training examples"

