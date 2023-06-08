# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:33:02 2022

@author: jkaneda

Re-counts the max number of time steps in the ACL dataset.
"""
import os
import numpy as np

basedir = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\interim'
subjects = os.listdir(os.path.join(basedir, 'ACL_FeatureMatrices'))
if '.DS_Store' in subjects: subjects.remove('.DS_Store')

max_timesteps = 0

for subj in subjects:
    featuremats = os.listdir(os.path.join(basedir, 'ACL_FeatureMatrices', subj))
    for featuremat in featuremats:
        data = np.load(os.path.join(basedir, 'ACL_FeatureMatrices', subj, featuremat))
        if data.shape[0] > max_timesteps:
            max_timesteps = data.shape[0]
            
print(f"Max number of time steps in ACL dataset: {max_timesteps}")