# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 01:20:18 2023

@author: jkaneda

Functions for transforming predictions back into 3D vector notation and 
in the global reference frame.

Essentially the inverse of the corresponding functions in "build_features_utils.py"
"""
#%% IMPORTS

import numpy as np

#%%

def get_3d_vec(mag_unitvec):
    """
    Puts the magnitude + unit vector representation back into standard 3D
    vector notation.
    

    Inputs:
        mag_unitvec: (numpy array) magnitude (first column) and 
        x,y,z unit vector (columns 2-4) (size: num_timesteps x 4)
    
    Returns:
        vector_3d: (numpy array) 3D vector (size: num_timesteps x 3)
    """
    
    vector_3d = mag_unitvec[:,1:4] * mag_unitvec[:,0].reshape(-1,1)
    
    return vector_3d

#%%

def get_pelvis_in_ground_transformation_matrices(stateTrajectory, model, num_timesteps):
    """
    Compute the transformation matrices for the pelvis frame in the ground frame.
    
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
        
        # Put into 4x4 transformation matrix style.
        G_T_P_final = np.vstack((np.hstack((G_R_P, p_GP)), [0,0,0,1]))
        
        # Add G_T_P for the current state to the list of matrices.
        transforms.append(G_T_P_final)
    
    return transforms