# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 16:54:40 2023

@author: jkaneda

Creates a dictionary of the "random" trials to use for upsampling in the ACL dataset.
Also creates a list of subject IDs + trials that only have one trial.
"""

#%% IMPORTS
import os
import random
import numpy as np

#%% PROCESS

outdict = dict()
singletriallist = []

BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload' # your base directory
SEED = 43
random.seed(SEED)

acl_tasks = ['_cutting','_dj','_lldj','_rldj','_unant_cut']

# Get ACL subjects list: use same as CS230 final report, but remove Subject 102913_315 from test set due to error in double jump task (repeat of lldj) 
subjects_acl =  os.listdir(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices'))
if '.DS_Store' in subjects_acl: subjects_acl.remove('.DS_Store')

for subj in subjects_acl:
    
    # Get list of all feature matrix files for given subject.
    feats_list = os.listdir(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj))
    # Make feature mat list lowercase.
    feats_list_lc = [mat.lower() for mat in feats_list if '100Hz' in mat]
    
    if len(feats_list_lc) != 15:
        
        for task in acl_tasks:
                
            task_count = sum(map(lambda x: task in x, feats_list_lc))
            task_filenames = [mat for mat in feats_list_lc if task in mat]
            
            if task_count == 2:
                
                # Randomly select which one of the task trials to repeat.
                task_to_repeat_ix = random.randint(0,1)
                
                # Add to dict.
                outdict[subj + task] = task_filenames[task_to_repeat_ix]
            
            if task_count == 1:
                
                singletriallist.append(subj + task)

# Save the dict
np.savez(os.path.join(BASE_DIR, 'references', 'ACL_TrialsToUpsample.npz'), **outdict)

# Save the list
with open(os.path.join(BASE_DIR, 'references', 'ACL_SingleTrials.txt'), 'w') as f:
    for trial in singletriallist:
        f.write(f"{trial}\n")

