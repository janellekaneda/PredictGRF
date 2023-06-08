# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:33:02 2022

@author: jkaneda

Checks which of the six tasks are missing, if any, in the ACL dataset for each subject.
"""
import os
import numpy as np

basedir = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\interim'
subjects = os.listdir(os.path.join(basedir, 'ACL_FeatureMatrices'))

tasks = ['_cutting','_dj','_lldj','_rldj','_unant_cut']
abnormal = []

for subj in subjects:
    featuremats = os.listdir(os.path.join(basedir, 'ACL_FeatureMatrices', subj))
    featuremats = [feature.lower() for feature in featuremats]
    for task in tasks:
        if not any([task in feature for feature in featuremats]):
            abnormal.append((subj + ': ' + task))

print(abnormal)