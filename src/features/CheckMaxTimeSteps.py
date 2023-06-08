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

if dataset == 'ACL':
    for subj in subjects:
        featuremats = os.listdir(os.path.join(basedir, (dataset + '_FeatureMatrices'), subj))
        for featuremat in featuremats:
            data = np.load(os.path.join(basedir, (dataset + '_FeatureMatrices'), subj, featuremat))
            if data.shape[0] > max_timesteps:
                max_timesteps = data.shape[0]
                
if dataset == 'OAGR':
    for subj in subjects:
        conditions = os.listdir(os.path.join(basedir, (dataset + '_FeatureMatrices'), subj))
        for condition in conditions:
            featuremats = os.listdir(os.path.join(basedir, (dataset + '_FeatureMatrices'), subj, condition))
            for featuremat in featuremats:
                data = np.load(os.path.join(basedir, (dataset + '_FeatureMatrices'), subj, condition, featuremat))
                if data.shape[0] > max_timesteps:
                    max_timesteps = data.shape[0]   
            
print(f"Max number of time steps in dataset: {max_timesteps}")