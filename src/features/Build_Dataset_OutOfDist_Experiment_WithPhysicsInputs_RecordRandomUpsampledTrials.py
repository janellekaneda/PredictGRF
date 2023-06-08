# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:26:21 2022

@author: jkaneda

EDIT: Sat Jan 7 2023: remove cutting and unant_cut from training sets.
"""

#%% IMPORTS

import os
import random
import numpy as np
from build_dataset_utils import *

#%% BATCH PROCESS

# # PARAMETERS # #
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload' # your base directory
MAX_TIMESTEPS = 67
MASK_VALUE = 999
NUM_INPUT_FEATS = 42
NUM_OUTPUT_FEATS = 16
NUM_OAGR_TRIALS = 100 # 20 steps x 5 conditions per subject
NUM_ACL_TRIALS_TRAIN = 9 # 3 trials x 3 tasks (dj, lldj, rldj), after upsampling
NUM_ACL_TRIALS_TEST_DEV = 15 # 3 trials x 5 tasks per subject, after upsampling
SEED = 43
# # # # # # # # # 

# Declare random seed.
random.seed(SEED)

# Condition / task names in each dataset for looping.
oagr_conds = ['baseline_TM1','eval_5deg1','eval_10deg1','eval_neg5deg1','eval_neg10deg1']
acl_tasks = ['_cutting','_dj','_lldj','_rldj','_unant_cut']
acl_tasks_train = ['_dj','_lldj','_rldj'] 

# # Get lists of train, dev, and test subject IDs for each dataset. # #
subjects_oagr = os.listdir(os.path.join(BASE_DIR, 'data', 'interim', 'OAGR_FeatureMatrices'))
if '.DS_Store' in subjects_oagr: subjects_oagr.remove('.DS_Store')

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

# Get ACL subjects list: use same as CS230 final report, but remove Subject 102913_315 from test set due to error in double jump task (repeat of lldj) 
subjects_acl =  os.listdir(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices'))
if '.DS_Store' in subjects_acl: subjects_acl.remove('.DS_Store')

train_subjs_acl_file = open(os.path.join(BASE_DIR, 'references', 'ACL_train_subjs.txt'), 'r')
dev_subjs_acl_file   = open(os.path.join(BASE_DIR, 'references', 'ACL_dev_subjs.txt'), 'r')
test_subjs_acl_file  = open(os.path.join(BASE_DIR, 'references', 'ACL_test_subjs.txt'), 'r')

train_subjs_acl = train_subjs_acl_file.read().split('\n')
dev_subjs_acl   = dev_subjs_acl_file.read().split('\n')
test_subjs_acl  = test_subjs_acl_file.read().split('\n')
train_subjs_acl.remove('') # remove trailing space
dev_subjs_acl.remove('')
test_subjs_acl.remove('')

#%%

# # Initialize train, dev, and test arrays with mask value, and dataset identifier lists. # #
num_train_ex_oagr = len(train_subjs_oagr) * NUM_OAGR_TRIALS
num_train_ex_acl = len(train_subjs_acl) * NUM_ACL_TRIALS_TRAIN
X_train = np.ones((num_train_ex_oagr + num_train_ex_acl, MAX_TIMESTEPS, NUM_INPUT_FEATS)) * MASK_VALUE
Y_train = np.ones((num_train_ex_oagr + num_train_ex_acl, MAX_TIMESTEPS, NUM_OUTPUT_FEATS)) * MASK_VALUE

num_dev_ex_oagr = len(dev_subjs_oagr) * NUM_OAGR_TRIALS
num_dev_ex_acl = len(dev_subjs_acl) * NUM_ACL_TRIALS_TEST_DEV
X_dev = np.ones((num_dev_ex_oagr + num_dev_ex_acl, MAX_TIMESTEPS, NUM_INPUT_FEATS)) * MASK_VALUE
Y_dev = np.ones((num_dev_ex_oagr + num_dev_ex_acl, MAX_TIMESTEPS, NUM_OUTPUT_FEATS)) * MASK_VALUE

num_test_ex_oagr = len(test_subjs_oagr) * NUM_OAGR_TRIALS
num_test_ex_acl = len(test_subjs_acl) * NUM_ACL_TRIALS_TEST_DEV
X_test = np.ones((num_test_ex_oagr + num_test_ex_acl, MAX_TIMESTEPS, NUM_INPUT_FEATS)) * MASK_VALUE
Y_test = np.ones((num_test_ex_oagr + num_test_ex_acl, MAX_TIMESTEPS, NUM_OUTPUT_FEATS)) * MASK_VALUE

# # Loop over each split dataset, and append accordingly. # #
# Init ix to sequentially add data to each data array.
train_ix_all = 0
dev_ix_all = 0
test_ix_all = 0

# Initiate dictionary to store randomly selected upsampled trials.
# (Relevant for ACL dataset, for tasks with two trials -- need to choose which one to upsample)
ACL_test_upsampledtrials = dict()

# Initialize a list to store all test set trials and the order.
test_set_trials = []


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
                print(f"Appending to test set, {subj}, {cond}, {feats_filename}, ix: {test_ix_all}")
                test_set_trials.append(f"Appending to test set, {subj}, {cond}, {feats_filename}, ix: {test_ix_all}")
                    
                X_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                Y_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                test_ix_all += 1
    
    #print(f"Finished {subj}")

# Next, loop over ACL dataset.
for subj in subjects_acl:
    
    # Get list of all feature matrix files for given subject.
    feats_list = os.listdir(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj))
    # Make feature mat list lowercase.
    feats_list_lc = [mat.lower() for mat in feats_list]
    
    # Make separate lists for training subjects (without cutting tasks).
    feats_list_train = [feat for feat in feats_list if '_cut' not in feat]
    feats_list_lc_train = [mat.lower() for mat in feats_list_train]
    
    if subj in train_subjs_acl: # only use feats_list_lc_train
        
        if len(feats_list_lc_train) == NUM_ACL_TRIALS_TRAIN: # append normally             
            
            for feats_filename in feats_list_train:
                
                # Load feature matrix.
                feats = np.load(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj, feats_filename), allow_pickle=True)
                
                # Get number of time steps for given trial.
                num_timesteps = feats.shape[0]
                
                # Extract X and Y (inputs and outputs).
                X_data = feats[:, 0:NUM_INPUT_FEATS] # shape = num timesteps x num input features
                Y_data = feats[:, NUM_INPUT_FEATS:(NUM_INPUT_FEATS+NUM_OUTPUT_FEATS)]
                
                # Append to data split set.
                X_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0) # add axis for trial num
                Y_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                train_ix_all += 1 # Increment index.
        
        else: # need to append additional trials with upsampling, retaining same task order
        
            for task in acl_tasks_train:
                    
                task_count = sum(map(lambda x: task in x, feats_list_lc_train))
                task_filenames = [mat for mat in feats_list_lc_train if task in mat]
                
                if task_count == 3: # append normally
                
                    for task_filename in task_filenames:
                        
                        # Load feature matrix.
                        feats = np.load(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj, task_filename), allow_pickle=True)
                        
                        # Get number of time steps for given trial.
                        num_timesteps = feats.shape[0]
                        
                        # Extract X and Y (inputs and outputs).
                        X_data = feats[:, 0:NUM_INPUT_FEATS] # shape = num timesteps x num input features
                        Y_data = feats[:, NUM_INPUT_FEATS:(NUM_INPUT_FEATS+NUM_OUTPUT_FEATS)]
                        
                        # Append to data split set.
                        X_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0) # add axis for trial num
                        Y_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        train_ix_all += 1 # Increment index.
            
                
                if task_count == 1: # repeat that task's single trial three times
                    
                    # Load feature matrix.
                    feats = np.load(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj, task_filenames[0]), allow_pickle=True)
                    
                    # Get number of time steps for given trial.
                    num_timesteps = feats.shape[0]
                    
                    # Extract X and Y (inputs and outputs).
                    X_data = feats[:, 0:NUM_INPUT_FEATS] # shape = num timesteps x num input features
                    Y_data = feats[:, NUM_INPUT_FEATS:(NUM_INPUT_FEATS+NUM_OUTPUT_FEATS)]
                    
                    # First, append that single trial.
                    
                    # Append to data split set.
                    X_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0) # add axis for trial num
                    Y_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                    train_ix_all += 1 # Increment index.
                    
                    # Then, upsample.
                    for upsample in range(2):
                    
                        # Append to data split set.
                        X_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0) # add axis for trial num
                        Y_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        train_ix_all += 1 # Increment index.
                
                
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
                        
                        # Append to data split set.
                        X_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0) # add axis for trial num
                        Y_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        train_ix_all += 1 # Increment index.
                
                    # Then, randomly select which one of the task trials to repeat.
                    task_to_repeat_ix = random.randint(0,1)
                    
                    # Load feature matrix.
                    feats = np.load(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj, task_filenames[task_to_repeat_ix]), allow_pickle=True)
                    
                    # Get number of time steps for given trial.
                    num_timesteps = feats.shape[0]
                    
                    # Extract X and Y (inputs and outputs).
                    X_data = feats[:, 0:NUM_INPUT_FEATS] # shape = num timesteps x num input features
                    Y_data = feats[:, NUM_INPUT_FEATS:(NUM_INPUT_FEATS+NUM_OUTPUT_FEATS)]
                    
                    # Append to data split set.
                    X_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0) # add axis for trial num
                    Y_train[train_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                    train_ix_all += 1 # Increment index.
    
    
    else: # dev or test
    
        if len(feats_list_lc) == NUM_ACL_TRIALS_TEST_DEV: # append normally             
            
            for feats_filename in feats_list:
                
                # Load feature matrix.
                feats = np.load(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj, feats_filename), allow_pickle=True)
                
                # Get number of time steps for given trial.
                num_timesteps = feats.shape[0]
                
                # Extract X and Y (inputs and outputs).
                X_data = feats[:, 0:NUM_INPUT_FEATS] # shape = num timesteps x num input features
                Y_data = feats[:, NUM_INPUT_FEATS:(NUM_INPUT_FEATS+NUM_OUTPUT_FEATS)]
                
                # Append to each data split set based on what group the current subject is in.
                if subj in dev_subjs_acl:
                    X_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                    Y_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                    dev_ix_all += 1
                
                if subj in test_subjs_acl:
                    print(f"Appending to test set, {subj}, {feats_filename}, ix: {test_ix_all}")
                    test_set_trials.append(f"Appending to test set, {subj}, {feats_filename}, ix: {test_ix_all}")
                    
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
                        if subj in dev_subjs_acl:
                            X_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                            Y_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                            dev_ix_all += 1
                        
                        if subj in test_subjs_acl:
                            print(f"Appending to test set, {subj}, {task_filename}, ix: {test_ix_all}")
                            test_set_trials.append(f"Appending to test set, {subj}, {task_filename}, ix: {test_ix_all}")
                            
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
                    if subj in dev_subjs_acl:
                        X_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                        Y_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        dev_ix_all += 1
                    
                    if subj in test_subjs_acl:
                        print(f"Appending to test set, {subj}, {task_filenames[0]}, ix: {test_ix_all}")
                        test_set_trials.append(f"Appending to test set, {subj}, {task_filenames[0]}, ix: {test_ix_all}")
                        
                        X_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                        Y_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        test_ix_all += 1
                    
                    # Then, upsample.
                    for upsample in range(2):
                    
                        # Append to each data split set based on what group the current subject is in.                        
                        if subj in dev_subjs_acl:
                            X_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                            Y_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                            dev_ix_all += 1
                        
                        if subj in test_subjs_acl:
                            print(f"Appending to test set, {subj}, {task_filenames[0]}, ix: {test_ix_all}")
                            test_set_trials.append(f"Appending to test set, {subj}, {task_filenames[0]}, ix: {test_ix_all}")
                            
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
                        if subj in dev_subjs_acl:
                            X_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                            Y_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                            dev_ix_all += 1
                        
                        if subj in test_subjs_acl:
                            print(f"Appending to test set, {subj}, {task_filename}, ix: {test_ix_all}")
                            test_set_trials.append(f"Appending to test set, {subj}, {task_filename}, ix: {test_ix_all}")
                            
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
                    if subj in dev_subjs_acl:
                        X_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                        Y_dev[dev_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        dev_ix_all += 1
                    
                    if subj in test_subjs_acl:
                        print(f"Appending to test set, {subj}, {task_filenames[task_to_repeat_ix]}, ix: {test_ix_all}")
                        test_set_trials.append(f"Appending to test set, {subj}, {task_filenames[task_to_repeat_ix]}, ix: {test_ix_all}")
                        

                        X_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(X_data, axis=0)
                        Y_test[test_ix_all, 0:num_timesteps, :] = np.expand_dims(Y_data, axis=0)
                        test_ix_all += 1
                        
                        # Add to dictionary.
                        ACL_test_upsampledtrials[subj] = task_filenames[task_to_repeat_ix]
                    
    #print(f"Finished {subj}")

# Save the train, dev, and test set arrays.
outpath = os.path.join(BASE_DIR, 'data', 'processed','not_normalized' ,'with_physicsinputs')
np.save(os.path.join(outpath, 'X_train.npy'), X_train)
np.save(os.path.join(outpath, 'Y_train.npy'), Y_train)
np.save(os.path.join(outpath, 'X_dev.npy'), X_dev)
np.save(os.path.join(outpath, 'Y_dev.npy'), Y_dev)
np.save(os.path.join(outpath, 'X_test.npy'), X_test)
np.save(os.path.join(outpath, 'Y_test.npy'), Y_test)

# Save the upsampled trial dictionary.
np.save(os.path.join(BASE_DIR, 'references', 'ACL_test_upsampledtrials.npy'), ACL_test_upsampledtrials)

# Write test set trials to file.
with open(os.path.join(BASE_DIR, 'references', 'test_set_trials.txt'), 'w') as f:
    for entry in test_set_trials:
        f.write(entry + '\n')

# Check that appended correctly...
assert train_ix_all == num_train_ex_oagr + num_train_ex_acl, "ERROR: train indices don't match total number of training examples"
assert dev_ix_all == num_dev_ex_oagr + num_dev_ex_acl, "ERROR: dev indices don't match total number of training examples"
assert test_ix_all == num_test_ex_oagr + num_test_ex_acl, "ERROR: test indices don't match total number of training examples"
