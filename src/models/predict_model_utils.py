# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:44:27 2022

@author: jkaneda

Contains functions for predicting on the given model.
"""

#%% IMPORTS

from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import os

#%%
def analyze_oagr(Y_pred_oagr, Y_true_oagr, Y_train_means, Y_train_stds, num_subjs, num_trials, data_type, loss_type, plot_savedir, MASK_VALUE=999):
    """
    Plots and calculates error metrics.
    oagr_data shape = (700, 168, 16)
    """
    # Store output names.
    out_node_names = ['R GRF mag','R GRF x','R GRF y','R GRF z',
                      'L GRF mag','L GRF x','L GRF y','L GRF z',
                      'R GRM mag','R GRM x','R GRM y','R GRM z',
                      'L GRM mag','L GRM x','L GRM y','L GRM z']
    
    # Init arrays to store error metrics.
    rmses = np.zeros((num_subjs,16))
    corrs = np.zeros((num_subjs,16))
    
    # loop over 7 test subjects, and get all steps per subj
    for s in range(num_subjs):
        
        Y_pred = Y_pred_oagr[num_trials*s:num_trials*(s+1), :, :] # shape = 100 x 168 x 16
        Y_true = Y_true_oagr[num_trials*s:num_trials*(s+1), :, :]
        
        # Identify the minimum where masking/padding started for the given subject over all trials for given task.
        min_timestep = np.min(np.nonzero(Y_true == MASK_VALUE)[1])
        
        # Create subplots for all the outputs for each subject.
        fig, axs = plt.subplots(4, 4, figsize=(15,15))
        fig.suptitle(f"{data_type}{loss_type}: OAGR Subject {s}")
        
        for f in range(16): # loop over all features
            
            # Subset Y_pred and Y_true based on min timesteps and given feature.
            Y_pred = Y_pred[:,0:min_timestep,f] # shape = 100, unmasked time range, 1 feature
            Y_true = Y_true[:,0:min_timestep,f]
        
            # Reverse normalize based on which dataset training means and stds.
            Y_pred = (Y_pred*Y_train_stds.T[f] + Y_train_means.T[f]) 
            Y_true = (Y_true*Y_train_stds.T[f] + Y_train_means.T[f])
            
            
            # # ERROR METRICS # # 
            # Calculate RMSE for each feature.
            rmses[s,f] = sqrt(mean_squared_error(Y_true.ravel(), Y_pred.ravel()))
        
            # Calculate corr for each feature.
            r = np.corrcoef(Y_true.ravel(), Y_pred.ravel())
            corr = r[0,1]
            if np.isnan(corr):
                corrs[s,f] = 0
            else:
                corrs[s,f] = corr
                
            
            # # PLOTTING # #
            # Add to subplots:
            if f < 4: row = 0
            elif f < 8: row = 1
            elif f < 12: row = 2
            else: row = 3
        
            axs[row, f%4].plot(np.mean(Y_pred,axis=0),'r')
            axs[row, f%4].plot(np.mean(Y_true,axis=0),'b')
            axs[row, f%4].set_title(out_node_names[f])
            if f == 0:
              axs[row, f%4].legend(['prediction','true'])
            if f == 3:
              axs[row, f%4].xtitle('timesteps')
              
        # Save figure for each subject.
        figname = 'oagr_subj' + str(s) + '.png'
        plt.savefig(os.path.join(plot_savedir, figname))
        
    return rmses, corrs


#%%

def analyze_acl(Y_pred_acl, Y_true_acl, Y_train_means, Y_train_stds, num_subjs, num_trials, data_type, loss_type, plot_savedir, MASK_VALUE=999):
    """
    Plots and calculates error metrics.
    oagr_data shape = (120, 168, 16)
    """
    # Store output names.
    out_node_names = ['R GRF mag','R GRF x','R GRF y','R GRF z',
                      'L GRF mag','L GRF x','L GRF y','L GRF z',
                      'R GRM mag','R GRM x','R GRM y','R GRM z',
                      'L GRM mag','L GRM x','L GRM y','L GRM z']
    
    acl_tasks = ['_cutting','_dj','_lldj','_rldj','_unant_cut']
    
    # Init arrays to store error metrics.
    rmses = np.zeros((num_subjs*5,16)) # 5 tasks
    corrs = np.zeros((num_subjs*5,16)) # 5 tasks
    metric_counter = 0
    
    # loop over test subjects, and get all steps per subj
    for s in range(num_subjs):
        
        # Subset data by subject
        Y_pred = Y_pred_acl[num_trials*5*s:num_trials*5*(s+1), :, :] # shape = 15 x 168 x 16
        Y_true = Y_true_acl[num_trials*5*s:num_trials*5*(s+1), :, :]
        
        for task in range(5): # 5 tasks
        
            # Subset data by task.
            Y_pred = Y_pred[num_trials*task:num_trials*(task+1),:,:] #  shape = 3 x 168 x 16
            Y_true = Y_true[num_trials*task:num_trials*(task+1),:,:]
        
        
            # Identify the minimum where masking/padding started for the given subject over all trials for given task.
            min_timestep = np.min(np.nonzero(Y_true == MASK_VALUE)[1])
            
            # Create subplots for all the outputs for each subject and task.
            fig, axs = plt.subplots(4, 4, figsize=(15,15))
            fig.suptitle(f"{data_type}{loss_type}: ACL Subject {s}, {acl_tasks[task]}")
            
            for f in range(16): # loop over all features
                
                # Subset Y_pred and Y_true based on min timesteps and given feature.
                Y_pred = Y_pred[:,0:min_timestep,f] # shape = 3, unmasked time range, 1 feature
                Y_true = Y_true[:,0:min_timestep,f]
            
                # Reverse normalize based on which dataset training means and stds.
                Y_pred = (Y_pred*Y_train_stds.T[f] + Y_train_means.T[f]) 
                Y_true = (Y_true*Y_train_stds.T[f] + Y_train_means.T[f])
                
                # # ERROR METRICS # # 
                # Calculate RMSE for each feature.
                rmses[metric_counter,f] = sqrt(mean_squared_error(Y_true.ravel(), Y_pred.ravel()))
            
                # Calculate corr for each feature.
                r = np.corrcoef(Y_true.ravel(), Y_pred.ravel())
                corr = r[0,1]
                if np.isnan(corr):
                    corrs[metric_counter,f] = 0
                else:
                    corrs[metric_counter,f] = corr
    
                
                # # PLOTTING # #
                # Add to subplots:
                if f < 4: row = 0
                elif f < 8: row = 1
                elif f < 12: row = 2
                else: row = 3
            
                axs[row, f%4].plot(np.mean(Y_pred,axis=0),'r')
                axs[row, f%4].plot(np.mean(Y_true,axis=0),'b')
                axs[row, f%4].set_title(out_node_names[f])
                if f == 0:
                  axs[row, f%4].legend(['prediction','true'])
                if f == 3:
                  axs[row, f%4].xtitle('timesteps')
            
            # Save figure for each subject and task.
            figname = 'acl_subj' + str(s) + '_' + acl_tasks[task] + '.png'
            plt.savefig(os.path.join(plot_savedir, figname))
            
            metric_counter += 1
        
    return rmses, corrs
