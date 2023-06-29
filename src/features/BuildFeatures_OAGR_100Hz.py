# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 20:41:00 2022

@author: jkaneda

Constructs input and output features for each step and step trial in the OAGR dataset.

See "build_features_utils.py" for greater explanation of the functions.

Re-run for data downsampled to 100 Hz.
Additional updates:
    - add output features for contact classification, total GRF, GRF distribution
    - remove pelvis-related IK features (no root info!)
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

# Add general OAGR model geometry to the OpenSim system.
geopath = os.path.join(BASE_DIR, 'data', 'raw', 'OAGR_Geometry', 'Geometry')
opensim.ModelVisualizer.addDirToGeometrySearchPaths(geopath)

# # List of our IK features of interest. # #
ikresnames = ['ankle_angle_l','ankle_angle_r',
              'hip_adduction_l','hip_adduction_r',
              'hip_flexion_l','hip_flexion_r',
              'hip_rotation_l','hip_rotation_r',
              'knee_adduction_l','knee_adduction_r',
              'subtalar_angle_l','subtalar_angle_r']

# # List of all OAGR subject IDs. # #
subjects = list(range(101,170,1)) # Numbers 101 to 169
subjects.remove(104) # No Subject 104 in dataset
subjects = [str(x) for x in subjects] # make strings for filenames

# # List of all gait types. # #
gait_types = ['baseline_TM1','eval_5deg1','eval_10deg1','eval_neg5deg1','eval_neg10deg1']

# # Specify starting and ending step numbers. # #
steps_start = 1
steps_end = 20
steps_list = [str(x) for x in list(range(steps_start, steps_end+1, 1))]

# # Keep track of max number of time steps. # #
max_num_timesteps = 0

# # Load in Excel spreadsheet of OAGR subject body masses and heights.
bm_h = pd.read_excel(os.path.join(BASE_DIR, 'references', 'OAGR_BodyMass_Height.xlsx'))

# Loop over all subjects, gait types, and steps:

for subj in subjects:
    
    # # Get subject's model. # #
    
    # Subject's model filepath.
    modelpath = os.path.join(BASE_DIR, 'data', 'raw', 'OAGR_DownSampledFiles', ('Subject_' + subj), (subj + '_scaled_WinbyDist.osim'))
    model = get_model('oagr', modelpath)
    
    # # Subject's body mass and height. # #
    bodymass = bm_h.loc[bm_h.Subject == (int(subj)), 'BodyMass'].values[0]
    height = bm_h.loc[bm_h.Subject == (int(subj)), 'Height'].values[0]
            
    for gait_type in gait_types:
        
        for step in steps_list:
            
            # # First, create IK results input features. # #
            
            # Specify IK results filepath.
            ikrespath = os.path.join(BASE_DIR, 'data', 'raw', 'OAGR_DownSampledFiles', ('Subject_' + subj), gait_type, ('results_ik_step' + step + '_Fs100.sto'))
            ik_features = get_ik_features(ikrespath, ikresnames)
            
            
            # # Calculate whole-body and foot-body center of mass 3D kinematics. # #
            
            # Get state trajectory and other info for duration of IK results.
            stateTrajectory, num_timesteps, time = get_state_trajectory(model, ikrespath)
            
            # Get whole-body and toe and heel body center of mass kinematics.
            com_pos, _, com_acc = get_com_kinematics(stateTrajectory, model, num_timesteps, time)
            ltoe_vel, rtoe_vel = get_body_velocity(stateTrajectory, model, num_timesteps, 'toes')
            lheel_vel, rheel_vel = get_body_velocity(stateTrajectory, model, num_timesteps, 'calcn')
            
            # Estimate force acting on whole-body center of mass.
            com_force = com_acc * bodymass
            
            
            # # Get ground reaction features, and normalize them. # #
            
            # Specify ground reaction loads filepaths.
            grpath = os.path.join(BASE_DIR, 'data', 'raw', 'OAGR_DownSampledFiles', ('Subject_' + subj), gait_type, ('forces_step' + step + '_Fs100.mot'))
            
            # 'skip' for leg parameter since affected leg does not affect file extraction order.
            grf_r, grf_l, grm_r, grm_l = get_gr_features('oagr', 'skip', grpath, bodymass, height)
            
            # Get total GRF
            grf_total = get_total_grf(grf_r, grf_l)
            
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
            
            grf_total_t = apply_transformation_matrices(transforms, grf_total)
            
            # # Split transformed vectors into their magnitude + 3D unit vector representations. # #
            com_force_feats = get_mag_unitvec(com_force_t)
            com_pos_feats   = get_mag_unitvec(com_pos_t)
            rheel_vel_feats = get_mag_unitvec(rheel_vel_t)
            rtoe_vel_feats  = get_mag_unitvec(rtoe_vel_t)
            lheel_vel_feats = get_mag_unitvec(lheel_vel_t)
            ltoe_vel_feats  = get_mag_unitvec(ltoe_vel_t)
            
            grf_r_feats     = get_mag_unitvec(grf_r_t)
            grf_l_feats     = get_mag_unitvec(grf_l_t)
            grm_r_feats     = get_mag_unitvec(grm_r_t)
            grm_l_feats     = get_mag_unitvec(grm_l_t)
            
            grf_total_feats = get_mag_unitvec(grf_total_t)
            
            # # Calculate step classification outputs. # #
            step_class_r_feats, step_class_l_feats = get_step_classification(grf_r_feats[:,0], grf_l_feats[:,0])
            
            # # Calculate force distribution outputs. # #
            force_dist_r_feats, force_dist_l_feats = get_force_distribution(grf_r, grf_l)
            assert np.any((force_dist_r_feats < 0) | (force_dist_r_feats > 1)) == 0, f"*** R DIST FEATS OUT OF BOUNDS: {subj}, {gait_type}, {step}: {force_dist_r_feats}"
            assert np.any((force_dist_l_feats < 0) | (force_dist_l_feats > 1)) == 0, f"*** l DIST FEATS OUT OF BOUNDS: {subj}, {gait_type}, {step}: {force_dist_l_feats}"
                
            #if np.min(force_dist_r_feats) < 0 or np.min(force_dist_l_feats) < 0 or np.max(force_dist_r_feats) > 1 or np.max(force_dist_l_feats) > 1:
                #print(f"{subj}, {gait_type}, {step}")
            
            #print(f"{subj}, {gait_type}, {step}: r: {force_dist_r_feats}, l: {force_dist_l_feats}")
            
            # # Finally, save concatenated feature matrix. # #
            outdir = os.path.join(BASE_DIR, 'data', 'interim', 'OAGR_FeatureMatrices', ('Subject_' + subj), gait_type)
            if not os.path.exists(outdir):
               os.makedirs(outdir)
            outfilename = ('Subject_' + subj + '_' + gait_type + '_step' + step + '_INPUT_OUTPUT_100Hz_RawDist.npy')
            outpath = os.path.join(outdir, outfilename)
            

            create_feature_matrix(ik_features, com_force_feats, com_pos_feats,
                                      rheel_vel_feats, rtoe_vel_feats,
                                      lheel_vel_feats, ltoe_vel_feats,
                                      grf_r_feats, grf_l_feats,
                                      grm_r_feats, grm_l_feats,
                                      grf_total_feats,
                                      step_class_r_feats,
                                      step_class_l_feats,
                                      force_dist_r_feats,
                                      force_dist_l_feats,
                                      outpath, save_to_file = True)
            
            # # Keep track of maximum number of time steps. # #
            if num_timesteps > max_num_timesteps:
                max_num_timesteps = num_timesteps
            
    
    print(f"Finished Subject {subj}")

print(f"Maximum number of time steps: {max_num_timesteps}")
                   