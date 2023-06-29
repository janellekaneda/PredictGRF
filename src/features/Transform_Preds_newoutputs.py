# -*- coding: utf-8 -*-
"""
Created on Wed Jan  11 18:19:00 2023

@author: jkaneda

Load model predictions, and 1) transform them back into the global reference
frame and 2) put back into 3D vector representation.

NOTE: checked previously, no upsampling done for ACL test subjects, so can
loop normally
"""

#%% IMPORTS

import os
import random
import numpy as np
from build_dataset_utils import *
from build_features_utils import *
from transform_preds_utils import *

#%% BATCH PROCESS

# # PARAMETERS # #
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload' # your base directory
MAX_TIMESTEPS = 112
MASK_VALUE = 999
NUM_OUTPUT_FEATS = 15 # back into 3D vec form, plus total GRF
NUM_OAGR_TRIALS = 100 # 20 steps x 5 conditions per subject
NUM_ACL_TRIALS_TEST_DEV = 15 # 3 trials x 5 tasks per subject, after upsampling
SEED = 43
# # # # # # # # # 
exp_type = 'with_physicsinputs'
use_transforms = True

# Declare random seed.
random.seed(SEED)

# Condition / task names in each dataset for looping.
oagr_conds = ['baseline_TM1','eval_5deg1','eval_10deg1','eval_neg5deg1','eval_neg10deg1']

# # Get lists of train, dev, and test subject IDs for each dataset. # #

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

#%%

# # Initialize transformed Y_test and Y_pred arrays with mask value. # #
num_test_ex_oagr = len(test_subjs_oagr) * NUM_OAGR_TRIALS
num_test_ex_acl = len(test_subjs_acl) * NUM_ACL_TRIALS_TEST_DEV
Y_true_transformed = np.ones((num_test_ex_oagr + num_test_ex_acl, MAX_TIMESTEPS, NUM_OUTPUT_FEATS)) * MASK_VALUE # shape: (trials, max timesteps, 12)
Y_pred_transformed = np.ones((num_test_ex_oagr + num_test_ex_acl, MAX_TIMESTEPS, NUM_OUTPUT_FEATS)) * MASK_VALUE

# # Load in Y_true and Y_pred data structs. (originals = pelvis reference frame, mag + unit vec representation) # #
Y_true = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'normalized', exp_type, 'Y_test_norm_100Hz_RawDist.npy'), allow_pickle=True) # shape: (trials, max timesteps, num outputs)
# Subset Y_true for cont
Y_true = Y_true[:,:,0:20]
Y_pred = np.load(os.path.join(BASE_DIR, 'models', exp_type, 'lstm_newoutputs_transformfix_rawdist_tunedparams', 'testset_predictions_cont.npy'), allow_pickle=True)

# Separate into OAGR and ACL datasets.
# =============================================================================
# Y_true_oagr = Y_true[0:700,:,:]
# Y_true_acl  = Y_true[700:805,:,:]
# Y_pred_oagr = Y_pred[0:700,:,:]
# Y_true_acl  = Y_pred[700:805,:,:]
# =============================================================================

# Load training means and stds to unnormalize.
Y_train_means = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'normalized', exp_type, 'Y_train_means_100Hz_RawDist.npy'), allow_pickle=True)
Y_train_stds = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'normalized', exp_type, 'Y_train_stds_100Hz_RawDist.npy'), allow_pickle=True)

# Only get the first 20 for cont outputs
Y_train_means = Y_train_means[:,0:20]
Y_train_stds = Y_train_stds[:,0:20]

# # Loop over each split dataset, and append accordingly. # #
# Init ix to sequentially add data to each data array.
test_ix_all = 0

# # Specify starting and ending step numbers for OAGR dataset. # #
steps_start = 1
steps_end = 20
#steps_list = [str(x) for x in list(range(steps_start, steps_end+1, 1))]
steps_list =[str(x) for x in [10,11,12,13,14,15,16,17,18,19,1,20,2,3,4,5,6,7,8,9]] # due to Python path list order

# Loop over OAGR dataset first.
for subj in test_subjs_oagr:
    
    # # Get subject's model. # #
    
    # Subject's model filepath.
    modelpath = os.path.join(BASE_DIR, 'data', 'raw', 'OAGR_DownSampledFiles', ('Subject_' + subj), (subj + '_scaled_WinbyDist.osim'))
    model = get_model('oagr', modelpath)

    for cond in oagr_conds:
        
        for step in steps_list:
            
            # # Isolate the non-masked values. # #
            Y_true_curr_mask = Y_true[test_ix_all,:,:].reshape(1,Y_true.shape[1], Y_true.shape[2]) # shape: (1 trial, max timesteps, 16)
            min_timestep = np.min(np.nonzero(Y_true_curr_mask == MASK_VALUE)[1])
            Y_true_curr_unmask = Y_true[test_ix_all,0:min_timestep,:] # shape: (min timesteps, 16)
            Y_pred_curr_unmask = Y_pred[test_ix_all,0:min_timestep,:]
            
            # Un-normalize predictions.
            Y_true_curr_unmask = (Y_true_curr_unmask * Y_train_stds) + Y_train_means
            Y_pred_curr_unmask = (Y_pred_curr_unmask * Y_train_stds) + Y_train_means
            
            # # Put the output features back into 3D vector notation. # #
            true_grf_r_3d = get_3d_vec(Y_true_curr_unmask[:,0:4]) # shape: (min timesteps, 3)
            true_grf_l_3d = get_3d_vec(Y_true_curr_unmask[:,4:8])                                       
            true_grm_r_3d = get_3d_vec(Y_true_curr_unmask[:,8:12])            
            true_grm_l_3d = get_3d_vec(Y_true_curr_unmask[:,12:16])
            true_grf_tot_3d = get_3d_vec(Y_true_curr_unmask[:,16:]) 
            
            pred_grf_r_3d = get_3d_vec(Y_pred_curr_unmask[:,0:4])
            pred_grf_l_3d = get_3d_vec(Y_pred_curr_unmask[:,4:8])                                       
            pred_grm_r_3d = get_3d_vec(Y_pred_curr_unmask[:,8:12])            
            pred_grm_l_3d = get_3d_vec(Y_pred_curr_unmask[:,12:16])  
            pred_grf_tot_3d = get_3d_vec(Y_pred_curr_unmask[:,16:])                                               
            
            
            if use_transforms:
                # # Get pelvis-in-ground transformation matrices (G_T_P). # #
                
                # Specify IK results filepath.
                ikrespath = os.path.join(BASE_DIR, 'data', 'raw', 'OAGR_DownSampledFiles', ('Subject_' + subj), cond, ('results_ik_step' + step + '_Fs100.sto'))
    
                
                # Get state trajectory and other info for duration of IK results.
                stateTrajectory, num_timesteps, time = get_state_trajectory(model, ikrespath)
                
                # Check that min_timestep and num_timesteps variables agree!
                assert min_timestep == num_timesteps, f"ERROR: timesteps don't agree for Subject {subj}, {cond}, step {step}"
    # =============================================================================
    #             if min_timestep != num_timesteps:
    #                 print(f"Subject {subj}, {cond}, step {step}, ix = {test_ix_all}, min_timestep = {min_timestep}, num_timesteps = {num_timesteps}")
    #                 print(ikrespath)
    # =============================================================================
    
                # Get transformation matrices.
                transforms = get_pelvis_in_ground_transformation_matrices(stateTrajectory, model, num_timesteps)
                    
                # # Transform back into the ground/global reference frame. # #
                true_grf_r_t = apply_transformation_matrices(transforms, true_grf_r_3d) # shape: (min_timesteps, 3)
                true_grf_l_t = apply_transformation_matrices(transforms, true_grf_l_3d)            
                true_grm_r_t = apply_transformation_matrices(transforms, true_grm_r_3d)
                true_grm_l_t = apply_transformation_matrices(transforms, true_grm_l_3d)   
                true_grf_tot_t = apply_transformation_matrices(transforms, true_grf_tot_3d) 
    
                pred_grf_r_t = apply_transformation_matrices(transforms, pred_grf_r_3d)
                pred_grf_l_t = apply_transformation_matrices(transforms, pred_grf_l_3d)            
                pred_grm_r_t = apply_transformation_matrices(transforms, pred_grm_r_3d)
                pred_grm_l_t = apply_transformation_matrices(transforms, pred_grm_l_3d)
                pred_grf_tot_t = apply_transformation_matrices(transforms, pred_grf_tot_3d) 
                
                # Place into big transformed datastructs (contains masked value)!
                Y_true_transformed[test_ix_all,0:min_timestep,:] = np.expand_dims(np.hstack([true_grf_r_t, true_grf_l_t, true_grm_r_t, true_grm_l_t, true_grf_tot_t]), axis=0)
                Y_pred_transformed[test_ix_all,0:min_timestep,:] = np.expand_dims(np.hstack([pred_grf_r_t, pred_grf_l_t, pred_grm_r_t, pred_grm_l_t, pred_grf_tot_t]), axis=0)
                
                # Don't forget to increment counter!
                test_ix_all += 1
            
            
            else:
                
                # Place into big transformed datastructs (contains masked value)!
                Y_true_transformed[test_ix_all,0:min_timestep,:] = np.expand_dims(np.hstack([true_grf_r_3d, true_grf_l_3d, true_grm_r_3d, true_grm_l_3d, true_grf_tot_3d]), axis=0)
                Y_pred_transformed[test_ix_all,0:min_timestep,:] = np.expand_dims(np.hstack([pred_grf_r_3d, pred_grf_l_3d, pred_grm_r_3d, pred_grm_l_3d, pred_grf_tot_3d]), axis=0)
                
                # Don't forget to increment counter!
                test_ix_all += 1
                

    print(f"Finished Subject {subj}")


# # Next, repeat process for ACL dataset. # #
for subj in test_subjs_acl:
    
    # # Get list of all subject's files. # #
    subjdir = os.path.join(BASE_DIR, 'data', 'raw', 'ACL_DownSampledFiles', subj)
    subj_allfiles = os.listdir(subjdir)
    
    # # Get subject's model. # #
    
    # Subject's model filepath.
    model_filename = [filename for filename in subj_allfiles if 'osim' in filename][0]
    modelpath = os.path.join(subjdir, model_filename)
    model = get_model('acl', modelpath)

    # Get full list, and correct Python order, of all trials for given subject.
    feats_list = os.listdir(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj))
    split1 = [feat.split(f"_{subj}_")[1] for feat in feats_list if '100Hz_RawDist' in feat]
    trialnames = [feat.split("_INPUT")[0] for feat in split1]   

    print(trialnames)    
    
    for trial in trialnames:
            
            # # Isolate the non-masked values. # #
            Y_true_curr_mask = Y_true[test_ix_all,:,:].reshape(1,Y_true.shape[1], Y_true.shape[2]) # shape: (1 trial, max timesteps, 16)
            min_timestep = np.min(np.nonzero(Y_true_curr_mask == MASK_VALUE)[1])
            Y_true_curr_unmask = Y_true[test_ix_all,0:min_timestep,:] # shape: (min timesteps, 16)
            Y_pred_curr_unmask = Y_pred[test_ix_all,0:min_timestep,:]
            
            # Un-normalize predictions.
            Y_true_curr_unmask = (Y_true_curr_unmask * Y_train_stds) + Y_train_means
            Y_pred_curr_unmask = (Y_pred_curr_unmask * Y_train_stds) + Y_train_means
            
            
            # # Put the output features back into 3D vector notation. # #
            true_grf_r_3d = get_3d_vec(Y_true_curr_unmask[:,0:4]) # shape: (min timesteps, 3)
            true_grf_l_3d = get_3d_vec(Y_true_curr_unmask[:,4:8])                                       
            true_grm_r_3d = get_3d_vec(Y_true_curr_unmask[:,8:12])            
            true_grm_l_3d = get_3d_vec(Y_true_curr_unmask[:,12:16])
            true_grf_tot_3d = get_3d_vec(Y_true_curr_unmask[:,16:]) 
            
            pred_grf_r_3d = get_3d_vec(Y_pred_curr_unmask[:,0:4])
            pred_grf_l_3d = get_3d_vec(Y_pred_curr_unmask[:,4:8])                                       
            pred_grm_r_3d = get_3d_vec(Y_pred_curr_unmask[:,8:12])            
            pred_grm_l_3d = get_3d_vec(Y_pred_curr_unmask[:,12:16]) 
            pred_grf_tot_3d = get_3d_vec(Y_pred_curr_unmask[:,16:])  
                                                  
            
            if use_transforms:
                # # Get pelvis-in-ground transformation matrices (G_T_P). # #
                
                # Specify IK results filepath.
                ikrespath = os.path.join(BASE_DIR, 'data', 'raw', 'ACL_DownSampledFiles', subj, ('Trimmed_' + trial + '_JCs_ik_updated_Fs100.sto'))
    
                # Get state trajectory and other info for duration of IK results.
                stateTrajectory, num_timesteps, time = get_state_trajectory(model, ikrespath)
                
                # Check that min_timestep and num_timesteps variables agree!
                assert min_timestep == num_timesteps, f"ERROR: timesteps don't agree for Subject {subj}, {trial}"
    
                # Get transformation matrices.
                transforms = get_pelvis_in_ground_transformation_matrices(stateTrajectory, model, num_timesteps)
                    
                # # Transform back into the ground/global reference frame. # #
                true_grf_r_t = apply_transformation_matrices(transforms, true_grf_r_3d) # shape: (min_timesteps, 3)
                true_grf_l_t = apply_transformation_matrices(transforms, true_grf_l_3d)            
                true_grm_r_t = apply_transformation_matrices(transforms, true_grm_r_3d)
                true_grm_l_t = apply_transformation_matrices(transforms, true_grm_l_3d)   
                true_grf_tot_t = apply_transformation_matrices(transforms, true_grf_tot_3d) 
    
                pred_grf_r_t = apply_transformation_matrices(transforms, pred_grf_r_3d)
                pred_grf_l_t = apply_transformation_matrices(transforms, pred_grf_l_3d)            
                pred_grm_r_t = apply_transformation_matrices(transforms, pred_grm_r_3d)
                pred_grm_l_t = apply_transformation_matrices(transforms, pred_grm_l_3d)
                pred_grf_tot_t = apply_transformation_matrices(transforms, pred_grf_tot_3d) 
                
                # Place into big transformed datastructs (contains masked value)!
                Y_true_transformed[test_ix_all,0:min_timestep,:] = np.expand_dims(np.hstack([true_grf_r_t, true_grf_l_t, true_grm_r_t, true_grm_l_t, true_grf_tot_t]), axis=0)
                Y_pred_transformed[test_ix_all,0:min_timestep,:] = np.expand_dims(np.hstack([pred_grf_r_t, pred_grf_l_t, pred_grm_r_t, pred_grm_l_t, pred_grf_tot_t]), axis=0)
                
                # Don't forget to increment counter!
                test_ix_all += 1
                
            else:
                # Place into big transformed datastructs (contains masked value)!
                Y_true_transformed[test_ix_all,0:min_timestep,:] = np.expand_dims(np.hstack([true_grf_r_3d, true_grf_l_3d, true_grm_r_3d, true_grm_l_3d, true_grf_tot_3d]), axis=0)
                Y_pred_transformed[test_ix_all,0:min_timestep,:] = np.expand_dims(np.hstack([pred_grf_r_3d, pred_grf_l_3d, pred_grm_r_3d, pred_grm_l_3d, pred_grf_tot_3d]), axis=0)
                
                # Don't forget to increment counter!
                test_ix_all += 1  

    print(f"Finished Subject {subj}")
    
# Save transformed Y_true and Y_pred.
outdir = os.path.join(BASE_DIR, 'models', exp_type, 'lstm_newoutputs_transformfix_rawdist_tunedparams')
if use_transforms:
    np.save(os.path.join(outdir, 'Y_true_cont_transformed.npy'), Y_true_transformed)
    np.save(os.path.join(outdir, 'Y_pred_cont_transformed.npy'), Y_pred_transformed)
else:
    np.save(os.path.join(outdir, 'Y_true_cont_3d.npy'), Y_true_transformed)
    np.save(os.path.join(outdir, 'Y_pred_cont_3d.npy'), Y_pred_transformed)    
    
# Check that indexed correctly
assert test_ix_all == Y_true_transformed.shape[0]
