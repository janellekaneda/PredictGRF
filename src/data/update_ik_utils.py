# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:44:48 2022

Functions to update pre 4.0 IK files.

@author: jkaneda
"""

#%% IMPORTS

import os
import opensim
import shutil

#%% FUNCTIONS

def copy_og_results(ogpath, ogfilename, newpath):
    """
    Copy original IK or GRF results to subject directory.
    
    Inputs:
        ogpath: path to original results file (no filename included)
        ogfilename: original filename (including file type ending)
        newpath: path to new destination to copy results to (no filename included)
    """
    shutil.copy2(os.path.join(ogpath, ogfilename), newpath)
    


def update_model(modelpath):
    """
    Load subject model to see if need to update socket connections (for post 4.0 format).
    
    Inputs:
        modepath: (str) full path to .osim model file
        
    Returns:
        model: (Osim object) .osim model file with updates if applied
    """
   
    model = opensim.Model(modelpath)
    model.initSystem()
    opensim.updateSocketConnecteesBySearch(model)
    model.initSystem()
    
    return model

def update_ik_results(model, ikrespaths, suffix="_updated"):
    """
    Use IK update function and write updated IK results file.
    
    Inputs:
        model: (Osim object) .osim model file
        ikrespaths: (list) list of full file paths to IK results files to update
        suffix: (str) ending to add to updated file names (before file type ending).
        Set as "" if want to overwrite files.
    
    Adapted from Opensim documentation.
    """
    
    opensim.updatePre40KinematicsFilesFor40MotionType(model, ikrespaths, suffix)