# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:18:25 2022

This module contains all the functions needed to build the input and output features.
We create an input-output numpy array per trial, and use these arrays to build the 
main model datasets after.

EDIT Fri Jan 6 2023: get_gr_features -- fix logic statement in line 395 so only double jump uses both plates
EDIT Sat Jan 7 2023: get_gr_features -- add lines to fix trials that use abnormal force plate numbers

@author: jkaneda
"""

#%% IMPORTS

import os
import opensim
import numpy as np
import scipy.interpolate as interpolate

from opensim_sto_reader import readMotionFile # Downloaded from https://gist.github.com/mitkof6

#%% Load subject's model and perform any modifications.

def get_model(dataset, modelpath):
    """
    Creates subject's model object from file, unlocks the mtp angle
    (for the center of mass kinematics NaN bug), and if the dataset is ACL, 
    updates socket connections for post 4.0 format.
    
    Inputs:
        dataset: (str) 'oagr' or 'acl'; specifies which dataset which determines which file columns to use
        modelpath: (str) full file path to .osim model file
        
    Returns:
        model: (OSim object) model for given subject
    """
    # Load model.
    model = opensim.Model(modelpath)
    model.initSystem()
    
    # Unlock mtp angle in model for COM kinematics NaN bug.
    coords = model.getCoordinateSet()
    coords.get('mtp_angle_r').setDefaultLocked(False)
    coords.get('mtp_angle_l').setDefaultLocked(False)
    
    # Update socket connections for ACL models.
    if dataset == 'acl':
        opensim.updateSocketConnecteesBySearch(model)
        model.initSystem()
    
    return model

#%% Adapted from OpenCap script: get all the states for each time step (added mtp unlocking)

def get_state_trajectory(model, ikrespath):
    """
    Get the state trajectory for the given inverse kinematics (IK) results.
    
    Inputs:
        model: (OSim object) model for given subject
        ikrespath: (str) full file path to IK results file (.sto, .mot type)
    
    Returns:
        stateTrajectory: (OSim object) contains state variables for all time steps in IK results
        num_timesteps: (scalar, int type) number of timesteps in IK results
        time: (numpy array) for all time step stamps in IK results
        
    Code to obtain state trajectory adapted from OpenCap processing utilities:
    https://github.com/stanfordnmbl/opencap-processing/blob/main/utilsKinematics.py
    """
    
    # Make time-series type table from IK results (coordinates).
    table = opensim.TimeSeriesTable(ikrespath)
    tableProcessor = opensim.TableProcessor(table)
    tableProcessor.append(opensim.TabOpUseAbsoluteStateNames()) # for pre 4.0/post 4.0 consistency
    time = np.asarray(table.getIndependentColumn()) # time from IK results
    time = time - time[0] # start time at 0 if not already
    num_timesteps = table.getNumRows()
    
    # Convert IK results to radians (which is needed for states trajectory).
    table = tableProcessor.processAndConvertToRadians(model)
    
    # Compute coordinate speeds and add to IK results table (needed for states
    # trajectory).
    Qs = table.getMatrix().to_numpy()
    Qds = np.zeros(Qs.shape)
    columnAbsoluteLabels = list(table.getColumnLabels()) # names of IK results
    for i, columnLabel in enumerate(columnAbsoluteLabels):
        spline = interpolate.InterpolatedUnivariateSpline(
            time, Qs[:,i], k=3)
        # Coordinate speeds
        splineD1 = spline.derivative(n=1)
        Qds[:,i] = splineD1(time)     
        # Add coordinate speeds to table.
        columnLabel_speed = columnLabel[:-5] + 'speed'
        table.appendColumn(
            columnLabel_speed, 
            opensim.Vector(Qds[:,i].flatten().tolist()))
    
    # Append missing muscle states to table.
    # Needed for StatesTrajectory.
    stateVariableNames = model.getStateVariableNames()
    stateVariableNamesStr = [
        stateVariableNames.get(i) for i in range(
            stateVariableNames.getSize())]
    existingLabels = table.getColumnLabels()
    for stateVariableNameStr in stateVariableNamesStr:
        if not stateVariableNameStr in existingLabels:
            vec_0 = opensim.Vector([0] * table.getNumRows())            
            table.appendColumn(stateVariableNameStr, vec_0)
                    
    # Set state trajectory.
    stateTrajectory = opensim.StatesTrajectory.createFromStatesTable(model, table)
    
    return stateTrajectory, num_timesteps, time

#%% Own code: get P_T_G transformation matrices at each state

# Get "ground frame in pelvis frame" transformation matrix (P_T_G) for each state.

def get_ground_in_pelvis_transformation_matrices(stateTrajectory, model, num_timesteps):
    """
    Compute the transformation matrices for the ground frame in the pelvis frame.
    
    Inputs:        
        stateTrajectory: (OSim object) contains state variables for all time steps in IK results
        model: (OSim object) model for given subject
        num_timesteps: (scalar, int type) number of timesteps in IK results
        
    Returns:
        transforms: (list, numpy arrays) list of numpy 4x4 transformation matrices at each state
    """

    bodies = model.getBodySet()
    pelvis = bodies.get('pelvis')
    transforms = [] # init to store all transformation matrices
    
    for i in range(num_timesteps):
        
        curr_state = stateTrajectory[i]
        model.realizeVelocity(curr_state) # realize to velocity for COM acceleration estimation
        
        # Get G_T_P transformation matrix from OpenSim built-in function.
        G_T_P = pelvis.getTransformInGround(curr_state)
        
        # Extract rotation matrix from G_T_P.
        G_R_P = G_T_P.R().toString() # put in Python object form
        G_R_P = np.matrix(G_R_P).reshape((3,3)) # put into numpy and reshape
        
        # Extract translation vector from G_T_P.
        p_GP = G_T_P.p().toString() # put into Python object form
        p_GP = p_GP.replace('~','') # remove leading '~' char
        p_GP = np.matrix(p_GP).T # put into numpy and transpose into ((3,1) shape)
        
        # Find the components that will make up P_T_G, inverse of G_T_P, without having to take the inverse.
        
        # Find P_R_G, which is the transpose of G_R_P.
        P_R_G = G_R_P.T
        
        # Find the inverse translation vector.
        p_PG = -1 * np.matmul(P_R_G, p_GP)
        
        # Construct target transformation matrix (P_T_G).
        P_T_G = np.vstack((np.hstack((P_R_G, p_PG)), [0,0,0,1]))
        
        # Add P_T_G for the current state to the list of matrices.
        transforms.append(P_T_G)
        
    return transforms
    
#%% Adapted from OpenCap script: get whole-model COM position, (velocity), and acceleration for input feature building.

def get_com_kinematics(stateTrajectory, model, num_timesteps, time):
    """
    Computes the position, velocity, and acceleration trajectories of the center of mass of the whole model.
    
    Inputs:
        stateTrajectory: (OSim object) contains state variables for all time steps in IK results
        model: (OSim object) model for given subject
        num_timesteps: (scalar, int type) number of timesteps in IK results
        time: (numpy array) for all time step stamps in IK results
        
    Returns:
        com_pos: (numpy array) 3D COM position for all time steps (size: num_timesteps x 3)
        com_vel: (numpy array) 3D COM velocity for all time steps (size: num_timesteps x 3)
        com_acc: (numpy array) 3D COM acceleration for all time steps (size: num_timesteps x 3)
        
    Code adapted from OpenCap processing utilities:
    https://github.com/stanfordnmbl/opencap-processing/blob/main/utilsKinematics.py
    """
    
    # Initialize vectors to store kinematics data.
    com_pos = np.zeros((num_timesteps, 3))
    com_vel = np.zeros((num_timesteps, 3))
    com_acc = np.zeros((num_timesteps, 3))
    
    # Loop over timesteps to realize to velocity stage at each state and compute COM position and velocity.
    for i in range(num_timesteps):
        curr_state = stateTrajectory[i]
        model.realizeVelocity(curr_state)
        com_pos[i,:] = model.calcMassCenterPosition(curr_state).to_numpy()
        com_vel[i,:] = model.calcMassCenterVelocity(curr_state).to_numpy()
        
    # Take derivative of COM velocity to calculate COM acceleration.
    for c in range(com_acc.shape[1]): # loop over x, y, z coordinates
        spline = interpolate.InterpolatedUnivariateSpline(
            time, com_vel[:,c], k=3)
        splineD1 = spline.derivative(n=1)
        com_acc[:,c] = splineD1(time)

    return com_pos, com_vel, com_acc 

#%% Own code: get toe/heel (L/R) center of mass velocity vectors.

def get_body_velocity(stateTrajectory, model, num_timesteps, body_name):
    """
    Get the body center of mass velocity for a specified body (left and right).
    
    Inputs:
        stateTrajectory: (OSim object) contains state variables for all time steps in IK results
        model: (OSim object) model for given subject
        num_timesteps: (scalar, int type) number of timesteps in IK results
        body_name: (str) body name, lowercase (do not specify left or right)
    
    Returns:
        body_l_vel: (numpy array) 3D velocity for left body center of mass for all time steps (size: num_timesteps x 3)
        body_r_vel: (numpy array) 3D velocity for right body center of mass for all time steps (size: num_timesteps x 3)  
    """
    
    # Get the left and right bodies of interest.
    bodies = model.getBodySet()
    body_l = bodies.get(body_name + '_l')
    body_r = bodies.get(body_name + '_r')
    
    # Get the mass centers of the bodies of interest.
    body_l_com = body_l.get_mass_center()
    body_r_com = body_r.get_mass_center()
    
    # Initialize vectors to store the 3D velocity vectors.
    body_l_vel = np.zeros((num_timesteps, 3))
    body_r_vel = np.zeros((num_timesteps, 3))
    
    for i in range(num_timesteps):
        curr_state = stateTrajectory[i]
        body_l_vel[i, :] = body_l.findStationVelocityInGround(curr_state, body_l_com).to_numpy()
        body_r_vel[i, :] = body_r.findStationVelocityInGround(curr_state, body_r_com).to_numpy()
        
    return body_l_vel, body_r_vel

#%% Apply transformation matrices to 3D vector inputs and output features

def apply_transformation_matrices(transforms, vector_3d):
    """
    Apply the corresponding transformation matrix to a set of 3D points at each state.
    
    Inputs:
        transforms: (list, numpy arrays) list of numpy 4x4 transformation matrices at each state (list length: num_timesteps, matrix size: 4 x 4)
        vector_3d: (numpy array) 3D vector to transform (size: num_timesteps x 3)
        
    Returns:
        transformed_vector_3d: (numpy array) transformed 3D vector (size: num_timesteps x 3)
        
    Operations adapted from: https://stackoverflow.com/questions/32900442/large-point-matrix-array-multiplication-in-numpy
    """

    # First, make the input vector_3d into homogeneous coordinate format: add a column of ones.
    ones_col = np.ones((vector_3d.shape[0],1))
    input_homocoord = np.append(vector_3d, ones_col, axis=1)
    
    # Put list of transformation matrices into numpy array.
    transforms_arr = np.asarray(transforms)
    
    # Use np.einsum to perform efficient multi-dimensional matrix multiplication.
    transformed_vec_w_homo = np.einsum('ijk,ik->ij', transforms_arr, input_homocoord)
    
    # Lastly, divide through the homogenous coordinate and output a 3D vector.
    transformed_vector_3d = transformed_vec_w_homo / transformed_vec_w_homo[:,3,None]
    transformed_vector_3d = transformed_vector_3d[:,0:3]
    
    return transformed_vector_3d

#%% Put 3D vector into a magnitude + 3D unit vector representation

def get_mag_unitvec(vector_3d):
    """
    Splits the input 3D vector into a vector of magnitudes and a 3D vector of
    unit vectors for all time steps (i.e., the length of the input vector).
    
    Inputs:
        vector_3d: (numpy array) 3D vector to split (size: num_timesteps x 3)
        
    Returns:
        mag_unitvec: (numpy array) corresponding magnitude (first column) and 
        x,y,z unit vector (columns 2-4) of input vector (size: num_timesteps x 4)
    """
    
    # Get vector magnitude at each time step.
    vec_norm = np.linalg.norm(vector_3d, axis=1)
    vec_norm = vec_norm.reshape(vec_norm.shape[0], 1)
    
    # Get unit vectors by dividing input vector by magnitude at each time step.
    vec_units = vector_3d / vec_norm
    
    # Combine the magntude and unit vectors into a single numpy array,
    # listing the magnitude first.
    mag_unitvec = np.concatenate((vec_norm, vec_units), axis=1)
    
    return mag_unitvec

#%% Get the inverse kinematics results of interest for the input features.

def get_ik_features(ikrespath, ikresnames):
    """
    Outputs a numpy array of the inverse kinematics results of interest,
    with columns in the order of ik_res_names.
    
    Inputs:
        ikrespath: (str) full file path to IK results file (.sto, .mot type)
        ikresnames: (list, strs) list of column labels in the IK results file
    
    Returns:
        ik_features: (numpy array) matrix of the IK results of interest (size: num_timesteps x len(ikresnames))
    """
    
    # Load IK results file and extract column labels and data.
    _, labels, data = readMotionFile(ikrespath)
    
    # Put data into np array form.
    data = np.asarray(data)
    
    # Get column indices of IK results of interest, in the order of ikresnames.
    target_ixs = [i for i,item in enumerate(labels) if item in set(ikresnames)]
    target_labels = []
    for ix in target_ixs: target_labels.append(labels[ix])
    
    target_dict = {target_labels[i]: target_ixs[i] for i in range(len(target_labels))}
    
    target_ix_sorted = []
    for key in sorted(target_dict): target_ix_sorted.append(target_dict[key])
    
    # Create output matrix with select columns from IK results data.
    ik_features = data[:, target_ix_sorted]
    
    return ik_features
    
#%% Get correct ground reaction output features based on data set type and trial type, etc.

def get_gr_features(dataset, leg, grpath, bodymass, height):
    """
    Outputs numpy arrays for right and left GRF and GRM,
    normalized by subject body weight (GRFs) or body weight * height (GRMs).
    
    Inputs:
        dataset: (str) 'oagr' or 'acl'; specifies which dataset which determines which file columns to use
        leg: (str, uppercase) 'R' or 'L'; specifies dominant leg for subject; only relevant for ACL dataset.
        If OAGR dataset, can provide other string (like 'skip' or 'none') to be sure that this info is not used.
        grpath: (str, all lowercase) full file path to ground reaction loads file (.sto or .mot type)
        bodymass: (scalar, flaot etc.) body mass of the subject, in units of kg
        height: (scalar, float etc.) height of the subject, in units of m
        
    Returns:
        grf_r, grf_l, grm_r, grm_: (numpy arrays) normalized ground reaction 
        loads (size: num_timesteps x 3)
        NOTE: splitting the vector into magnitude and its unit vector is done
        in a separate function
    """
    
    # Make path all lowercase (only needed for ACL dataset).
    grpath = os.path.normcase(grpath)
    
    # Load ground reaction loads file and extract column labels and data.
    _, labels, data = readMotionFile(grpath)
    
    # Put data into np array form.
    data = np.asarray(data)
    
    # Get number of timesteps.
    num_timesteps = data.shape[0]
    
    # Add ground reaction data based on dataset and task type:
    if dataset =='oagr':
        
        grf_r_ix = [labels.index('ground_force_vx'), labels.index('ground_force_vy'), labels.index('ground_force_vz')]
        grf_l_ix = [labels.index('1_ground_force_vx'), labels.index('1_ground_force_vy'), labels.index('1_ground_force_vz')]
        grm_r_ix = [labels.index('ground_torque_x'), labels.index('ground_torque_y'), labels.index('ground_torque_z')]
        grm_l_ix = [labels.index('1_ground_torque_x'), labels.index('1_ground_torque_y'), labels.index('1_ground_torque_z')]
        
        # Put corresponding columns into each output's matrix.
        grf_r = data[:,grf_r_ix]
        grf_l = data[:,grf_l_ix]
        grm_r = data[:,grm_r_ix]
        grm_l = data[:,grm_l_ix]
    
    elif dataset == 'acl':
        
        # For drop jump: right leg on force plate 1, left leg on force plate 2
        if '_dj' in grpath:
            grf_r_ix = [labels.index('1_ground_force_vx'), labels.index('1_ground_force_vy'), labels.index('1_ground_force_vz')]
            grf_l_ix = [labels.index('2_ground_force_vx'), labels.index('2_ground_force_vy'), labels.index('2_ground_force_vz')]
            grm_r_ix = [labels.index('1_ground_torque_x'), labels.index('1_ground_torque_y'), labels.index('1_ground_torque_z')]
            grm_l_ix = [labels.index('2_ground_torque_x'), labels.index('2_ground_torque_y'), labels.index('2_ground_torque_z')]
        
            # Put corresponding columns into each output's matrix.
            grf_r = data[:,grf_r_ix]
            grf_l = data[:,grf_l_ix]
            grm_r = data[:,grm_r_ix]
            grm_l = data[:,grm_l_ix]
            
            # Fix trials that use force plates 2 and 3 instead (right leg on 2, left leg on 3).
            grf_r_y = grf_r[:,1]
            grf_l_y = grf_l[:,1]
            grf_r_y = np.where(grf_r_y < 1, 0, grf_r_y) # zero out small signals
            grf_l_y = np.where(grf_l_y < 1, 0, grf_l_y)
            if np.count_nonzero(grf_r_y) < int(num_timesteps * 0.25) or np.count_nonzero(grf_l_y) < int(num_timesteps * 0.25):
                print("Using force plates 2 and 3.")
                grf_r_ix = [labels.index('2_ground_force_vx'), labels.index('2_ground_force_vy'), labels.index('2_ground_force_vz')]
                grf_l_ix = [labels.index('3_ground_force_vx'), labels.index('3_ground_force_vy'), labels.index('3_ground_force_vz')]
                grm_r_ix = [labels.index('2_ground_torque_x'), labels.index('2_ground_torque_y'), labels.index('2_ground_torque_z')]
                grm_l_ix = [labels.index('3_ground_torque_x'), labels.index('3_ground_torque_y'), labels.index('3_ground_torque_z')]
    
                # Put corresponding columns into each output's matrix.
                grf_r = data[:,grf_r_ix]
                grf_l = data[:,grf_l_ix]
                grm_r = data[:,grm_r_ix]
                grm_l = data[:,grm_l_ix]

            
        # For all other tasks, only force plate 2 was used (single leg)
        elif "cut" in grpath: # for cutting tasks, depends on which is dominant leg
            if leg == 'R':
                grf_r_ix = [labels.index('2_ground_force_vx'), labels.index('2_ground_force_vy'), labels.index('2_ground_force_vz')]
                grm_r_ix = [labels.index('2_ground_torque_x'), labels.index('2_ground_torque_y'), labels.index('2_ground_torque_z')]
                
                # Put corresponding columns into each output's matrix.
                grf_r = data[:,grf_r_ix]
                grm_r = data[:,grm_r_ix]
                
                # Fix trials that use force plate 1.
                grf_r_y = grf_r[:,1]
                grf_r_y = np.where(grf_r_y < 1, 0, grf_r_y)
                if np.count_nonzero(grf_r_y) < int(num_timesteps * 0.25):
                    print("Using force plate 1 for right leg.")
                    grf_r_ix = [labels.index('1_ground_force_vx'), labels.index('1_ground_force_vy'), labels.index('1_ground_force_vz')]
                    grm_r_ix = [labels.index('1_ground_torque_x'), labels.index('1_ground_torque_y'), labels.index('1_ground_torque_z')]
                    
                    # Put corresponding columns into each output's matrix.
                    grf_r = data[:,grf_r_ix]
                    grm_r = data[:,grm_r_ix]
                
                # For the other leg, make all zeros.
                grf_l = np.zeros(grf_r.shape)
                grm_l = np.zeros(grm_r.shape) # technically all same shape
            elif leg == 'L':
                grf_l_ix = [labels.index('2_ground_force_vx'), labels.index('2_ground_force_vy'), labels.index('2_ground_force_vz')]
                grm_l_ix = [labels.index('2_ground_torque_x'), labels.index('2_ground_torque_y'), labels.index('2_ground_torque_z')]
                
                # Put corresponding columns into each output's matrix.
                grf_l = data[:,grf_l_ix]
                grm_l = data[:,grm_l_ix]
                
                # Fix trials that use force plate 1.
                grf_l_y = grf_l[:,1]
                grf_l_y = np.where(grf_l_y < 1, 0, grf_l_y)
                if np.count_nonzero(grf_l_y) < int(num_timesteps * 0.25):
                    print("Using force plate 1 for left leg.")
                    grf_l_ix = [labels.index('1_ground_force_vx'), labels.index('1_ground_force_vy'), labels.index('1_ground_force_vz')]
                    grm_l_ix = [labels.index('1_ground_torque_x'), labels.index('1_ground_torque_y'), labels.index('1_ground_torque_z')]
                    
                    # Put corresponding columns into each output's matrix.
                    grf_l = data[:,grf_l_ix]
                    grm_l = data[:,grm_l_ix]
                
                # For the other leg, make all zeros.
                grf_r = np.zeros(grf_l.shape)
                grm_r = np.zeros(grm_l.shape) # technically all same shape
            else:
                print("Cutting task logical error")
       
        elif '_rldj' in grpath:
            grf_r_ix = [labels.index('2_ground_force_vx'), labels.index('2_ground_force_vy'), labels.index('2_ground_force_vz')]
            grm_r_ix = [labels.index('2_ground_torque_x'), labels.index('2_ground_torque_y'), labels.index('2_ground_torque_z')]
            
            # Put corresponding columns into each output's matrix.
            grf_r = data[:,grf_r_ix]
            grm_r = data[:,grm_r_ix]
            
            # Fix trials that use force plate 1.
            grf_r_y = grf_r[:,1]
            grf_r_y = np.where(grf_r_y < 1, 0, grf_r_y)
            if np.count_nonzero(grf_r_y) < int(num_timesteps * 0.25):
                print("Using force plate 1 for right leg.")
                grf_r_ix = [labels.index('1_ground_force_vx'), labels.index('1_ground_force_vy'), labels.index('1_ground_force_vz')]
                grm_r_ix = [labels.index('1_ground_torque_x'), labels.index('1_ground_torque_y'), labels.index('1_ground_torque_z')]
                
                # Put corresponding columns into each output's matrix.
                grf_r = data[:,grf_r_ix]
                grm_r = data[:,grm_r_ix]
            
            # For the other leg, make all zeros.
            grf_l = np.zeros(grf_r.shape)
            grm_l = np.zeros(grm_r.shape) # technically all same shape
            
        elif '_lldj' in grpath:
            grf_l_ix = [labels.index('2_ground_force_vx'), labels.index('2_ground_force_vy'), labels.index('2_ground_force_vz')]
            grm_l_ix = [labels.index('2_ground_torque_x'), labels.index('2_ground_torque_y'), labels.index('2_ground_torque_z')]
            
            # Put corresponding columns into each output's matrix.
            grf_l = data[:,grf_l_ix]
            grm_l = data[:,grm_l_ix]
            
            # Fix trials that use force plate 1.
            grf_l_y = grf_l[:,1]
            grf_l_y = np.where(grf_l_y < 1, 0, grf_l_y)
            if np.count_nonzero(grf_l_y) < int(num_timesteps * 0.25):
                print("Using force plate 1 for left leg.")
                grf_l_ix = [labels.index('1_ground_force_vx'), labels.index('1_ground_force_vy'), labels.index('1_ground_force_vz')]
                grm_l_ix = [labels.index('1_ground_torque_x'), labels.index('1_ground_torque_y'), labels.index('1_ground_torque_z')]
                
                # Put corresponding columns into each output's matrix.
                grf_l = data[:,grf_l_ix]
                grm_l = data[:,grm_l_ix]
            
            # For the other leg, make all zeros.
            grf_r = np.zeros(grf_l.shape)
            grm_r = np.zeros(grm_l.shape) # technically all same shape           
        
        else:
            print('acl logical error')
        
    else:
        print('error: check dataset name')
    
    # Normalize.
    grf_r = grf_r / (bodymass * 9.8) # body mass * gravity = weight
    grf_l = grf_l / (bodymass * 9.8)
    grm_r = grm_r / (bodymass * 9.8 * height)
    grm_l = grm_l / (bodymass * 9.8 * height)
    
    return grf_r, grf_l, grm_r, grm_l
   
#%% Create and save whole feature numpy arrays.

def create_feature_matrix(ik_features, com_force_feats, com_pos_feats,
                          rheel_vel_feats, rtoe_vel_feats,
                          lheel_vel_feats, ltoe_vel_feats,
                          grf_r_feats, grf_l_feats,
                          grm_r_feats, grm_l_feats,
                          outpath, save_to_file = True):
    
    """
    Concatenates input and output features into a single numpy array,
    and saves it to a .npy file to the specified path. The combined feature matrix
    will have shape: number of time steps x (number of inputs + number of outputs).
    
    Inputs:
        ik_features: (numpy array) matrix of the IK results of interest (size: num_timesteps x len(ikresnames))
        com_force_feats, com_pos_feats: (numpy array) magnitude + 3D unit vector for force acting on and position of the whole-body center of mass
        rheel_vel_feats, rtoe_vel_feats, lheel_vel_feats, ltoe_vel_feats: (numpy array) magnitude + 3D unit vector for right and left heel and toe model body center of masses
        grf_r_feats, grf_l_feats, grm_r_feats, grm_l_feats: (numpy array) magnitude + 3D unit vector for right and left GRFs and GRMs (output features)
        outpath: (str) full file path (including file name) for saving the concatenated structure to
        save_to_file: (bool) whether or not to save the concatenated structure to file
        
    Returns:
        None
    """
    
    # Stack the inputs and outputs column-wise, in order of parameters:
    feature_matrix = np.hstack((ik_features, com_force_feats, com_pos_feats,
                              rheel_vel_feats, rtoe_vel_feats,
                              lheel_vel_feats, ltoe_vel_feats,
                              grf_r_feats, grf_l_feats,
                              grm_r_feats, grm_l_feats))
    
    # Replace NaNs (caused by dividing by zero when making unit vectors) with zeros.
    feature_matrix = np.nan_to_num(feature_matrix)
    
    # Save to file.
    if save_to_file:
        np.save(outpath, feature_matrix)    
    