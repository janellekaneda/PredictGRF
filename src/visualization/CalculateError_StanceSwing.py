# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:38:44 2023

@author: jkaneda

Calculate specified error metric for stance and swing legs.
NOTE: data is in 3D vector form (12 output features).
"""
#%% IMPORTS

import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error

#%% BATCH PROCESS

# # PARAMETERS # #
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload' # your base directory
MASK_VALUE = 999
NUM_OAGR_TRIALS = 100 # 20 steps x 5 conditions per subject
NUM_ACL_TRIALS_TEST_DEV = 15 # 3 trials x 5 tasks per subject, after upsampling
# # # # # # # # # 
# Toggle these!
exp_type = 'without_physicsinputs'
error_metric = 'rmse'
#error_metric = 'corr'
norm_rmse = True

# # Load data for given experiment type. # #
Y_true_all = np.load(os.path.join(BASE_DIR, 'models', exp_type, 'transformed_preds', 'Y_true_3d.npy'), allow_pickle=True)
Y_pred_all = np.load(os.path.join(BASE_DIR, 'models', exp_type, 'transformed_preds', 'Y_pred_3d.npy'), allow_pickle=True)

Y_true_oagr = Y_true_all[0:700,:,:]
Y_pred_oagr = Y_pred_all[0:700,:,:]
Y_true_acl  = Y_true_all[700:805,:,:]
Y_pred_acl  = Y_pred_all[700:805,:,:]

# # Condition / task names in each dataset for looping. # #
oagr_conds = ['baseline_TM1','eval_5deg1','eval_10deg1','eval_neg5deg1','eval_neg10deg1']
#acl_conds = ['_cutting: 0 1 2','_dj: 3 4 5','_lldj: 6 7 8','_rldj: 9 10 11','_unant_cut: 12 13 14']

# # Get lists of test subject IDs for each dataset. # #

# Instead of new split, use same train:dev:test split from milestone report to better compare previous and new model:
test_subjs_oagr_file  = open(os.path.join(BASE_DIR, 'references', 'OAGR_test_subjs.txt'), 'r')
test_subjs_oagr  = test_subjs_oagr_file.read().split('\n')
test_subjs_oagr.remove('')
#test_subjs_oagr = [('Subject_' + id) for id in test_subjs_oagr_list]

# Get ACL subjects list: use same as CS230 final report, but remove Subject 102913_315 from test set due to error in double jump task:
test_subjs_acl_file  = open(os.path.join(BASE_DIR, 'references', 'ACL_test_subjs.txt'), 'r')
test_subjs_acl  = test_subjs_acl_file.read().split('\n')
test_subjs_acl.remove('')
test_subjs_acl = sorted(test_subjs_acl) # need to sort because not already sorted, and in order from Build_Dataset script


# # Load in affected/dominant leg info. # #
# OAGR:
oagr_xlsx = pd.read_excel(os.path.join(BASE_DIR, 'references', 'OAGR_BodyMass_Height_leg.xlsx'))
# ACL:
acl_xlsx = pd.ExcelFile(os.path.join(BASE_DIR, 'references', 'ACL_CompiledLeg_BodyMassHeight.xlsx'))
acl_leg_info = pd.read_excel(acl_xlsx, 'Leg')

# # Iniitialize lists to store all data to calculate error metric over. # #

true_stance_walk = [] # Append an array of size [3 planes (x,y,z), num timesteps]
true_swing_walk = []
pred_stance_walk = []
pred_swing_walk = []

true_stance_cut = []
true_swing_cut = []
pred_stance_cut = [] 
pred_swing_cut = []

true_stance_dj = [] # no swing leg for double jump task
pred_stance_dj = []

true_stance_slj = []
true_swing_slj = []
pred_stance_slj = []
pred_swing_slj = []

#%%
# # Loop through OAGR dataset first. # # (should be same order as "Transform_Preds.npy script")
for s, subj in enumerate(test_subjs_oagr):
    
    # Get affected leg.
    leg = oagr_xlsx.loc[oagr_xlsx.Subject == (int(subj)), 'AffectedLimb'].values[0]
    
    # Gather all 100 walking trials for the given subject.
    Y_true_subj = Y_true_oagr[NUM_OAGR_TRIALS*s:NUM_OAGR_TRIALS*(s+1), :, :]
    Y_pred_subj = Y_pred_oagr[NUM_OAGR_TRIALS*s:NUM_OAGR_TRIALS*(s+1), :, :]
    
    for trial in range(NUM_OAGR_TRIALS): # loop through each trial, and find the minimum number of time steps (aka unmasked).
        
        Y_true_curr = Y_true_subj[trial, :, :] # shape = (max time steps, 12 outputs)
        Y_pred_curr = Y_pred_subj[trial, :, :]
        
        min_timestep = np.min(np.where(Y_true_curr[:,0] == MASK_VALUE)[0]) # only need to check along one of the outputs
        
        Y_true = Y_true_curr[0:min_timestep, :] # shape = (min time steps for given trial, 12 outputs)
        Y_pred = Y_pred_curr[0:min_timestep, :]

        # Add to stance and swing arrays.
        true_grf_r = Y_true[:, 0:3]
        true_grf_l = Y_true[:, 3:6]
        pred_grf_r = Y_pred[:, 0:3]
        pred_grf_l = Y_pred[:, 3:6]
        
        if leg == 'R':
            true_stance_walk.append(true_grf_r)
            true_swing_walk.append(true_grf_l)
            pred_stance_walk.append(pred_grf_r)
            pred_swing_walk.append(pred_grf_l)
        elif leg == 'L':
            true_stance_walk.append(true_grf_l)
            true_swing_walk.append(true_grf_r)
            pred_stance_walk.append(pred_grf_l)
            pred_swing_walk.append(pred_grf_r)
            
        else: print("Error: cannot read leg value for {subj}")


# # Repeat for ACL dataset. # #
for s, subj in enumerate(test_subjs_acl):
    
    # Get dominant leg.
    leg = acl_leg_info.loc[acl_leg_info.SubjectID == subj, 'Leg'].values[0]
    
    # Gather all 15 trials for the given subject.
    Y_true_subj = Y_true_acl[NUM_ACL_TRIALS_TEST_DEV*s:NUM_ACL_TRIALS_TEST_DEV*(s+1), :, :]
    Y_pred_subj = Y_pred_acl[NUM_ACL_TRIALS_TEST_DEV*s:NUM_ACL_TRIALS_TEST_DEV*(s+1), :, :]

    for trial in range(NUM_ACL_TRIALS_TEST_DEV): # loop through each trial, and find the minimum number of time steps (aka unmasked).
        
        Y_true_curr = Y_true_subj[trial, :, :] # shape = (max time steps, 12 outputs)
        Y_pred_curr = Y_pred_subj[trial, :, :]
        
        min_timestep = np.min(np.where(Y_true_curr[:,0] == MASK_VALUE)[0]) # only need to check along one of the outputs
        
        Y_true = Y_true_curr[0:min_timestep, :] # shape = (min time steps for given trial, 12 outputs)
        Y_pred = Y_pred_curr[0:min_timestep, :]

        # Add to stance and swing arrays. (we know tasks were appended in alphabetical order, with exactly three trials per 5 tasks for all test set subjects, no upsampling)
        true_grf_r = Y_true[:, 0:3]
        true_grf_l = Y_true[:, 3:6]
        pred_grf_r = Y_pred[:, 0:3]
        pred_grf_l = Y_pred[:, 3:6]

        if trial == 0 or trial == 1 or trial == 2 or trial == 12 or trial == 13 or trial == 14: # cutting and unant cutting
            if leg == 'R':
                true_stance_cut.append(true_grf_r)
                true_swing_cut.append(true_grf_l)
                pred_stance_cut.append(pred_grf_r)
                pred_swing_cut.append(pred_grf_l)
            elif leg == 'L':
                true_stance_cut.append(true_grf_l)
                true_swing_cut.append(true_grf_r)
                pred_stance_cut.append(pred_grf_l)
                pred_swing_cut.append(pred_grf_r)
                
            else: print("Error: cannot read leg value for {subj}")
        
        elif trial == 3 or trial == 4 or trial == 5: # double jump
            true_stance_dj.append(true_grf_r)
            true_stance_dj.append(true_grf_l)
            pred_stance_dj.append(pred_grf_r)
            pred_stance_dj.append(pred_grf_l)
        
        elif trial == 6 or trial == 7 or trial == 8: # LLDJ
            true_stance_slj.append(true_grf_l)
            true_swing_slj.append(true_grf_r)
            pred_stance_slj.append(pred_grf_l)
            pred_swing_slj.append(pred_grf_r)
            
        elif trial == 9 or trial == 10 or trial == 11: # RLDJ
            true_stance_slj.append(true_grf_r)
            true_swing_slj.append(true_grf_l)
            pred_stance_slj.append(pred_grf_r)
            pred_swing_slj.append(pred_grf_l)
        
        else: print("Error: index not in range for ACL dataset")
        
#%% Calculate errors!

row_labels = ['stance GRF x','stance GRF y','stance GRF z',
              'swing GRF x','swing GRF y','swing GRF z']

column_labels = ['walking', 'cutting', 'double leg jump', 'single leg jump']

error_table = np.zeros(shape=(len(row_labels), len(column_labels)) )

# walking stance
for axis in range(3):
    if error_metric == 'rmse':
        if norm_rmse:
            error_table[axis, 0] = mean_squared_error(np.vstack(true_stance_walk)[:,axis], np.vstack(pred_stance_walk)[:,axis], squared=False) / (np.max(np.vstack(true_stance_walk)[:,axis]) - np.min(np.vstack(true_stance_walk)[:,axis]))
        else:
            error_table[axis, 0] = mean_squared_error(np.vstack(true_stance_walk)[:,axis], np.vstack(pred_stance_walk)[:,axis], squared=False)
    elif error_metric == 'corr':
        error_table[axis, 0] = np.corrcoef(np.vstack(true_stance_walk)[:,axis], np.vstack(pred_stance_walk)[:,axis])[0,1]
# walking swing
for axis in range(3):
    if error_metric == 'rmse':
        if norm_rmse:
            error_table[axis+3, 0] = mean_squared_error(np.vstack(true_swing_walk)[:,axis], np.vstack(pred_swing_walk)[:,axis], squared=False) / (np.max(np.vstack(true_swing_walk)[:,axis]) - np.min(np.vstack(true_swing_walk)[:,axis]))
        else:
            error_table[axis+3, 0] = mean_squared_error(np.vstack(true_swing_walk)[:,axis], np.vstack(pred_swing_walk)[:,axis], squared=False)
    elif error_metric == 'corr':
        error_table[axis+3, 0] = np.corrcoef(np.vstack(true_swing_walk)[:,axis], np.vstack(pred_swing_walk)[:,axis])[0,1]

# cutting stance
for axis in range(3):
    if error_metric == 'rmse':
        if norm_rmse:
            error_table[axis, 1] = mean_squared_error(np.vstack(true_stance_cut)[:,axis], np.vstack(pred_stance_cut)[:,axis], squared=False) / (np.max(np.vstack(true_stance_cut)[:,axis]) - np.min(np.vstack(true_stance_cut)[:,axis]))
        else:
            error_table[axis, 1] = mean_squared_error(np.vstack(true_stance_cut)[:,axis], np.vstack(pred_stance_cut)[:,axis], squared=False)
    elif error_metric == 'corr':
        error_table[axis, 1] == np.corrcoef(np.vstack(true_stance_cut)[:,axis], np.vstack(pred_stance_cut)[:,axis])[0,1]
# cutting swing
for axis in range(3):
    if error_metric == 'rmse':
        if norm_rmse:
            error_table[axis+3, 1] = mean_squared_error(np.vstack(true_swing_cut)[:,axis], np.vstack(pred_swing_cut)[:,axis], squared=False) / (np.max(np.vstack(true_swing_cut)[:,axis]) - np.min(np.vstack(true_swing_cut)[:,axis]))
        else:
            error_table[axis+3, 1] = mean_squared_error(np.vstack(true_swing_cut)[:,axis], np.vstack(pred_swing_cut)[:,axis], squared=False)
    elif error_metric == 'corr':
        error_table[axis+3, 1] = np.corrcoef(np.vstack(true_swing_cut)[:,axis], np.vstack(pred_swing_cut)[:,axis])[0,1]

# double leg jump
for axis in range(3):
    if error_metric == 'rmse':
        if norm_rmse:
            error_table[axis, 2] = mean_squared_error(np.vstack(true_stance_dj)[:,axis], np.vstack(pred_stance_dj)[:,axis], squared=False) / (np.max(np.vstack(true_stance_dj)[:,axis]) - np.min(np.vstack(true_stance_dj)[:,axis]))
        else:
            error_table[axis, 2] = mean_squared_error(np.vstack(true_stance_dj)[:,axis], np.vstack(pred_stance_dj)[:,axis], squared=False)
    elif error_metric == 'corr':
        error_table[axis, 2] = np.corrcoef(np.vstack(true_stance_dj)[:,axis], np.vstack(pred_stance_dj)[:,axis])[0,1]
for axis in range(3):
    error_table[axis+3, 2] = 0 # placeholder

# slj stance
for axis in range(3):
    if error_metric == 'rmse':
        if norm_rmse:
            error_table[axis, 3] = mean_squared_error(np.vstack(true_stance_slj)[:,axis], np.vstack(pred_stance_slj)[:,axis], squared=False) / (np.max(np.vstack(true_stance_slj)[:,axis]) - np.min(np.vstack(true_stance_slj)[:,axis]))
        else:
            error_table[axis, 3] = mean_squared_error(np.vstack(true_stance_slj)[:,axis], np.vstack(pred_stance_slj)[:,axis], squared=False)
    elif error_metric == 'corr':
        error_table[axis, 3] = np.corrcoef(np.vstack(true_stance_slj)[:,axis], np.vstack(pred_stance_slj)[:,axis])[0,1]
# slj swing
for axis in range(3):
    if error_metric == 'rmse':
        if norm_rmse:
            error_table[axis+3, 3] = mean_squared_error(np.vstack(true_swing_slj)[:,axis], np.vstack(pred_swing_slj)[:,axis], squared=False) / (np.max(np.vstack(true_swing_slj)[:,axis]) - np.min(np.vstack(true_swing_slj)[:,axis]))
        else:    
            error_table[axis+3, 3] = mean_squared_error(np.vstack(true_swing_slj)[:,axis], np.vstack(pred_swing_slj)[:,axis], squared=False)
    elif error_metric == 'corr':
        error_table[axis+3, 3] = np.corrcoef(np.vstack(true_swing_slj)[:,axis], np.vstack(pred_swing_slj)[:,axis])[0,1]
        

# Output!
error_table_out = pd.DataFrame(error_table, index=row_labels, columns=column_labels)
if norm_rmse:
    error_table_out.to_excel(os.path.join(BASE_DIR, 'reports', ('norm_' + error_metric + '_' + exp_type + '.xlsx') ))
else:
    error_table_out.to_excel(os.path.join(BASE_DIR, 'reports', (error_metric + '_' + exp_type + '.xlsx') ))
