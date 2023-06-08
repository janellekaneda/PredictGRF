# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:56:12 2022

@author: jkaneda

This script updates the IK result files from the ACL dataset to be in post 4.0 format.
"""

#%% IMPORTS
import os
from update_ik_utils import *

#%% LOOP OVER DATA

# Specify base data directory.
basedatadir = 'W:\Julie ACL project data'
# Specify base results directory.
resultsdir = os.path.join('W:\OA_GaitRetraining','Janelle','CS230','ForUpload','data','raw','ACL_DownSampledFiles')

# Specify age groups.
#age_group = ['Soccer ACL 10-12 yo', 'Soccer ACL 14-18 yo']
age_group = ['Soccer ACL 10-12 yo']
#age_group = ['Soccer ACL 14-18 yo']

# Run for only one subject? Input Subject ID as str. Make sure to only run
# the corresponding subject and age group too.
run_one_subj = 1
subj_id = '040114_100'

# Count number of viable subjects.
total_subjs_num = 0

# List of subjects with !=18 trials.
abnormal_subjs = []

for age in age_group:
    
    # Set subject groups based on age group.
    if age == 'Soccer ACL 10-12 yo':
        #subj_groups = ['Control','Heatwave','Monsoon']
        subj_groups = ['Monsoon']
    else:
        subj_groups = ['Control','Gunn','MA','Pinewood']
        #subj_groups = ['Gunn','MA','Pinewood'] # leave out control group for now bc different directory structure
        #subj_groups = ['Control']
        
    for subj_group in subj_groups:
        
        # Get path for subject group.
        subj_group_path = os.path.join(basedatadir, age, 'Pre', subj_group)
        
        if not run_one_subj:
            # Get list of all individual subject folder names.
            subjs_oglist = next(os.walk(subj_group_path))[1]
            
            # Remove subjects with no "post" data (weren't processed fully) and other extraneous conditions.
            no_post_strs = [' - No post',' - no post', 'needs processed', 'z']
            subjs = [subj_num for subj_num in subjs_oglist if not any(delete_str in subj_num for delete_str in no_post_strs)]
            
            # Other subject list cleaning:
            if subj_group == 'Heatwave':
                subjs.remove('090514_530 - bad data') # remove bad data
                subjs.remove('Corey') # remove extraneous folder
            if age == 'Soccer ACL 14-18 yo' and subj_group == 'Control':
                subjs.remove('030315_430') # no proccesed data
        else:
            subjs = [subj_id]
            
        total_subjs_num += len(subjs)

        for subj in subjs:
            
            # Make subject folder in results directory.
            subj_path = os.path.join(resultsdir, subj)
            if os.path.exists(subj_path) and not run_one_subj:
                pass # data already processed
            else:
                #os.mkdir(subj_path) # don't make if already exists
            
                # Get list of IK results files for the given subject.
                if subj == '032514_430':
                    subj_allikfiles_path = os.path.join(subj_group_path, subj, 'OpenSim ouput', 'NEW Not Fixed', 'IK', 'IK_w_modelJCs')
                elif age == 'Soccer ACL 14-18 yo' and subj_group == 'Control':
                    subj_allikfiles_path = os.path.join(subj_group_path, subj, 'OpenSim output', 'IK', 'IK_w_modelJCs')
                else:
                    subj_allikfiles_path = os.path.join(subj_group_path, subj, 'OpenSim output', 'NEW Not Fixed', 'IK', 'IK_w_modelJCs')
                
                subj_allikfiles = os.listdir(subj_allikfiles_path)
                
                trial_names_og = [x for x in subj_allikfiles if x.startswith('Trimmed')]
                
                # Remove trials with errors.
                trial_names = [trial for trial in trial_names_og if not any(error_str in trial for error_str in ['error'])]
            
                # List subjects with != 18 trials.
                if len(trial_names) != 18:
                    abnormal_subjs.append([str(len(trial_names)) +': ' + age + ' ' + subj_group + ' ' + subj])
                    
                # Update subject model if needed.
                if age == 'Soccer ACL 10-12 yo':
                    if subj == '032514_430':
                        modelpath = os.path.join(subj_group_path, subj, 'OpenSim ouput', 'NEW Not Fixed', (subj + '_scaled_NEW not fixed - JC.osim'))
                    elif subj == '032114_330':
                        modelpath = os.path.join(subj_group_path, subj, 'OpenSim output', 'NEW Not Fixed', (subj + '_scaled_NEW - not fixed - JC.osim'))
                    elif subj == '032514_600' or subj == '032614_400':
                        modelpath = os.path.join(subj_group_path, subj, 'OpenSim output', 'NEW Not Fixed', (subj + '_scaled - JC.osim'))
                    elif subj == '040114_100':
                        modelpath = os.path.join(subj_group_path, subj, 'OpenSim output', 'NEW Not Fixed', ('041114_100_scaled_NEW not fixed - JC.osim'))
                    else:
                        modelpath = os.path.join(subj_group_path, subj, 'OpenSim output', 'NEW Not Fixed', (subj + '_scaled_NEW not fixed - JC.osim'))
                else: # 14-18 yo group
                    if subj_group == 'MA' or subj_group == 'Pinewood':
                        if subj == '010215_1230':
                            modelpath = os.path.join(subj_group_path, subj, 'OpenSim output', 'NEW Not Fixed', (subj + '_scaled_JC.osim'))
                        elif subj == '121314_1100' or subj == '121314_930':
                            modelpath = os.path.join(subj_group_path, subj, 'OpenSim output', 'NEW Not Fixed', (subj + ' ' + ' _scaled_JC.osim'))
                        #else subj == '011015_1100' or subj == '011015_1400' or subj == '010315_1400' or subj == '010515_100' or subj == '010515_230':
                        else:
                             modelpath = os.path.join(subj_group_path, subj, 'OpenSim output', 'NEW Not Fixed', (subj + ' _scaled_JC.osim'))
                    elif subj_group == 'Control':
                        if subj == '010315_1230':
                            modelpath = os.path.join(subj_group_path, subj, 'OpenSim output', (subj + '_scaled_not fixed_JC.osim'))
                        elif subj == '103013_130':
                            modelpath = os.path.join(subj_group_path, subj, 'OpenSim output', ('subject_' + subj + '_scaled_JC.osim'))
                        else:
                            modelpath = os.path.join(subj_group_path, subj, 'OpenSim output', ('subject_' + subj + '_scaled_not fixed_JC.osim'))
                    else:
                        modelpath = os.path.join(subj_group_path, subj, 'OpenSim output', 'NEW Not Fixed', (subj + '_scaled_JC.osim'))
                
                model = update_model(modelpath)
                
                # Store full paths of all copied trials.
                ikrespaths_copied = []
                
                for trial in trial_names:
                    
                    # Copy IK results file to subject directory.
                    copy_og_results(subj_allikfiles_path, trial, subj_path)
                    # Add copied IK file path to list.
                    ikrespaths_copied.append(os.path.join(subj_path, trial))
                    
                # Update all IK result files for given subject.
                update_ik_results(model, ikrespaths_copied, suffix="_updated")

# Print total number of viable subjects.
print(f"Total number of viable subjects = {total_subjs_num}")
# Print abnormal subjects.
print("Subjects with abnormal number of IK trials:")
print(abnormal_subjs)

