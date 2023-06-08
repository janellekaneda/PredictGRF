# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 10:05:00 2022

@author: jkaneda

Constructs input and output features for each step and step trial in the ACL dataset.

See "build_features_utils.py" for greater explanation of the functions.

EDIT: Fri Jan 6 2023 -- re-run LLDJ and RLDJ trials with updated feature utils.
"""

#%% IMPORTS
import os
import opensim
import pandas as pd
from build_features_utils import *

#%% BATCH PROCESS

# # PARAMETERS # #
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload' # your base directory
# # # # # # # # # 

# Add general ACL model geometry to the OpenSim system.
geopath = os.path.join(BASE_DIR, 'data', 'raw', 'ACL_Geometry', 'Geometry')
opensim.ModelVisualizer.addDirToGeometrySearchPaths(geopath)

# # List of our 18 IK features of interest. # #
ikresnames = ['ankle_angle_l','ankle_angle_r',
              'hip_adduction_l','hip_adduction_r',
              'hip_flexion_l','hip_flexion_r',
              'hip_rotation_l','hip_rotation_r',
              'knee_adduction_l','knee_adduction_r',
              'pelvis_list','pelvis_rotation','pelvis_tilt',
              'pelvis_tx','pelvis_ty','pelvis_tz',
              'subtalar_angle_l','subtalar_angle_r']

# # List of all ACL subject IDs. # #
subjects = os.listdir(os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices'))
if '.DS_Store' in subjects: subjects.remove('.DS_Store')

# For getting IDs from "raw" folder
# =============================================================================
# subjects = os.listdir(os.path.join(BASE_DIR, 'data', 'raw', 'ACL_DownSampledFiles'))
# subjects.remove('.DS_Store')
# subjects.remove('040414_1000') # missing body weight and mass
# subjects.remove('090114_10') # missing demographic info; might be '090114_1000'
# subjects.remove('040114_100') # missing tasks
# subjects.remove('103013_130') # missing tasks
# subjects.remove('111213_315') # missing tasks
# 
# =============================================================================

# # Keep track of max number of time steps. # #
max_num_timesteps = 0

# # Load in Excel spreadsheet of ACL subject dominant leg, body mass and height info.
xlsx = pd.ExcelFile(os.path.join(BASE_DIR, 'references', 'ACL_CompiledLeg_BodyMassHeight.xlsx'))
leg_info = pd.read_excel(xlsx, 'Leg')
bm_h = pd.read_excel(xlsx, 'BodyMassHeight')


# Loop over all subjects:

for subj in subjects:
    
    # # Get list of all subject's files. # #
    subjdir = os.path.join(BASE_DIR, 'data', 'raw', 'ACL_DownSampledFiles', subj)
    subj_allfiles = os.listdir(subjdir)
    
    # # Get subject's model. # #
    model_filename = [filename for filename in subj_allfiles if 'osim' in filename][0]
    modelpath = os.path.join(subjdir, model_filename)
    model = get_model('acl', modelpath)
    
    # # Subject's body mass, height, and dominant leg. # #
    bodymass = bm_h.loc[bm_h.SubjectID == subj, 'BodyMass'].values[0]
    height = bm_h.loc[bm_h.SubjectID == subj, 'Height'].values[0]
    leg = leg_info.loc[leg_info.SubjectID == subj, 'Leg'].values[0]
            
    # # Loop over all tasks and trials for current subject. # #
    
    # Get list of trial names (number of trials and tasks is different for some subjects).
    downsampled_files = [filename for filename in subj_allfiles if 'Fs60' in filename]
    split_filenames = [substring.split('grf') for substring in downsampled_files if 'grf' in substring] # each GRF file has a corresponding IK file
    trialnames = [trialname[0][:-1][8:] for trialname in split_filenames] # Remove "Trimmed_" and trailing "_"
    
    # ONLY RUN FOR LLDJ AND RLDJ #
    trialnames = [trial for trial in trialnames if 'lldj' in trial.lower() or 'rldj' in trial.lower()]
    
    for trial in trialnames:
        
        # Check if feature matrix for the given trial already exists.
        outdir = os.path.join(BASE_DIR, 'data', 'interim', 'ACL_FeatureMatrices', subj)
        if not os.path.exists(outdir):
           os.makedirs(outdir)
        outfilename = ('Subject_' + subj + '_' + trial + '_INPUT_OUTPUT.npy')
        outpath = os.path.join(outdir, outfilename)
        
        #if not os.path.exists(outpath): # want to rewrite LLDJ and RLDJ trials
        
        # # First, create IK results input features. # #
        
        # Specify IK results filepath.
        ikrespath = os.path.join(BASE_DIR, 'data', 'raw', 'ACL_DownSampledFiles', subj, ('Trimmed_' + trial + '_JCs_ik_updated_Fs60.sto'))
        ik_features = get_ik_features(ikrespath, ikresnames)
        
        
        # # Calculate whole-body and foot-body center of mass 3D kinematics. # #
        
        # Get state trajectory and other info for duration of IK results.
        stateTrajectory, num_timesteps, time = get_state_trajectory(model, ikrespath)
        
        # Get whole-body and toe and heel body ccenter of mass kinematics.
        com_pos, _, com_acc = get_com_kinematics(stateTrajectory, model, num_timesteps, time)
        ltoe_vel, rtoe_vel = get_body_velocity(stateTrajectory, model, num_timesteps, 'toes')
        lheel_vel, rheel_vel = get_body_velocity(stateTrajectory, model, num_timesteps, 'calcn')
        
        # Estimate force acting on whole-body center of mass.
        com_force = com_acc * bodymass
        
        
        # # Get ground reaction features, and normalize them. # #
        
        # Specify ground reaction loads filepaths.
        grpath = os.path.join(BASE_DIR, 'data', 'raw', 'ACL_DownSampledFiles', subj, ('Trimmed_' + trial + '_grf_Fs60.mot'))
        
        # Input dominant leg for second parameter.
        grf_r, grf_l, grm_r, grm_l = get_gr_features('acl', leg, grpath, bodymass, height)
        
        
        # # Transform the 3D vector features to be in the pelvis reference frame. # #
        
        # Get transformation matrices at each time step.
        transforms = get_ground_in_pelvis_transformation_matrices(stateTrajectory, model, num_timesteps)
        
        # Apply transformation matrices:
        com_force_t = apply_transformation_matrices(transforms, com_force)
        com_pos_t   = apply_transformation_matrices(transforms, com_pos)
        rheel_vel_t = apply_transformation_matrices(transforms, rheel_vel)
        rtoe_vel_t  = apply_transformation_matrices(transforms, rtoe_vel)
        lheel_vel_t = apply_transformation_matrices(transforms, lheel_vel)
        ltoe_vel_t  = apply_transformation_matrices(transforms, ltoe_vel)
        
        grf_r_t = apply_transformation_matrices(transforms, grf_r)
        grf_l_t = apply_transformation_matrices(transforms, grf_l)
        grm_r_t = apply_transformation_matrices(transforms, grm_r)
        grm_l_t = apply_transformation_matrices(transforms, grm_l)
        
        # # Split transformed vectors into their magnitude + 3D unit vector representations. # #
        com_force_feats = get_mag_unitvec(com_force_t)
        com_pos_feats   = get_mag_unitvec(com_pos_t)
        rheel_vel_feats = get_mag_unitvec(rheel_vel_t)
        rtoe_vel_feats  = get_mag_unitvec(rtoe_vel_t)
        lheel_vel_feats = get_mag_unitvec(lheel_vel_t)
        ltoe_vel_feats  = get_mag_unitvec(ltoe_vel_t)
        
        grf_r_feats     = get_mag_unitvec(grf_r)
        grf_l_feats     = get_mag_unitvec(grf_l)
        grm_r_feats     = get_mag_unitvec(grm_r)
        grm_l_feats     = get_mag_unitvec(grm_l)
        
        # # Finally, save concatenated feature matrix. # #
        create_feature_matrix(ik_features, com_force_feats, com_pos_feats,
                                  rheel_vel_feats, rtoe_vel_feats,
                                  lheel_vel_feats, ltoe_vel_feats,
                                  grf_r_feats, grf_l_feats,
                                  grm_r_feats, grm_l_feats,
                                  outpath, save_to_file = True)
        
        # # Keep track of maximum number of time steps. # #
        if num_timesteps > max_num_timesteps:
            max_num_timesteps = num_timesteps
    
        print(f"Finished Subject {subj}, {trial}")
        
# =============================================================================
#         else:
#             continue
# =============================================================================
            
    
print(f"Maximum number of time steps: {max_num_timesteps}")
                   