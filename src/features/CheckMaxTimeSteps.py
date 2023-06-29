# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:33:02 2022

@author: jkaneda

Re-counts the max number of time steps in each dataset.
"""
import os
import numpy as np

dataset = 'ACL'

basedir = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\interim'
subjects = os.listdir(os.path.join(basedir, (dataset + '_FeatureMatrices')))
if '.DS_Store' in subjects: subjects.remove('.DS_Store')

max_timesteps = 0
min_timesteps = 112
min_feat = 'holder'

if dataset == 'ACL':
    for subj in subjects:
        featuremats = os.listdir(os.path.join(basedir, (dataset + '_FeatureMatrices'), subj))
        featuremats = [x for x in featuremats if '100Hz' in x]
        for featuremat in featuremats:
            data = np.load(os.path.join(basedir, (dataset + '_FeatureMatrices'), subj, featuremat))
            if data.shape[0] > max_timesteps:
                max_timesteps = data.shape[0]
            if data.shape[0] < min_timesteps:
                min_timesteps = data.shape[0]
                min_feat = featuremat
            
            for i in range(36):
                #print(f"{i}:{np.max(data[:,i])}")
                if np.max(data[:,i]) > 1000:
                    print(f"{i}, {featuremat}, {np.max(data[:,i])}")
                
            #print(data.shape[0])
            
            # debugging
            if np.min(data[:,58:]) < 0 or np.max(data[:,58:]) > 1:
                print(subj, featuremat)
                
if dataset == 'OAGR':
    for subj in subjects:
        conditions = os.listdir(os.path.join(basedir, (dataset + '_FeatureMatrices'), subj))
        for condition in conditions:
            featuremats = os.listdir(os.path.join(basedir, (dataset + '_FeatureMatrices'), subj, condition))
            featuremats = [x for x in featuremats if '100Hz' in x]
            for featuremat in featuremats:
                data = np.load(os.path.join(basedir, (dataset + '_FeatureMatrices'), subj, condition, featuremat))
                if data.shape[0] > max_timesteps:
                    max_timesteps = data.shape[0]   
                if data.shape[0] < min_timesteps:
                    min_timesteps = data.shape[0]
                    min_feat = featuremat
                    
                for i in range(36):
                    #print(f"{i}:{np.max(data[:,i])}")
                    if np.max(data[:,i]) > 1000:
                        print(f"{i}, {featuremat}, {np.max(data[:,i])}")
                
                #print(data.shape[0])
                    
                # debugging
                if np.min(data[:,58:]) < 0 or np.max(data[:,58:]) > 1:
                    print(subj, condition, featuremat)
            
print(f"Max number of time steps in dataset: {max_timesteps}")
print(f"Min number of time steps in dataset: {min_timesteps}, {min_feat}")