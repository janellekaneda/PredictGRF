# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:30:03 2022

This module contains a function to create the train:dev:test split
for building the main model datasets.

@author: jkaneda
"""

#%% IMPORTS

import math
import random

#%% Get train - dev - test split

def get_train_dev_test_split(train_split, subjects_list, seed):
    """
    Get a list of subjects for train, dev, and test based on the given data split.
    If the number of subjects is not evenly divisible for the given data split, 
    the best rounded value will be used.
    
    Inputs:
        train_split: (float) decimal 0 to 1 of desired training split (e.g. .8).
        The dev and test split will automatically be equal and compromise the rest of the split (e.g. .1 and .1).
        subjects_list: (list, strs) list of subject IDs as strings
        seed: (int) seed number to use for reproducible random shuffling
    Returns:
        train_subjs: (list, strs) list of train subject IDs
        dev_subjs: (list, strs) list of dev subject IDs
        test_subjs: (list, strs) list of test subject IDs
    """
    
    # Randomize the input list of subjects.
    random.Random(seed).shuffle(subjects_list)
    
    # Get total number of subjects.
    total_subjs = len(subjects_list)
    
    # Determine number of training subjects.
    
    # Try both lower and upper values of the uneven split.
    train_subjs_floor = math.floor(train_split * total_subjs)
    train_subjs_ceil  = math.ceil(train_split * total_subjs)
    
    devtest_floor = total_subjs - train_subjs_floor
    devtest_ceil  = total_subjs - train_subjs_ceil
    
    if devtest_floor%2 == 0: # the candidate dev/test split is even
        train_subjs_num = train_subjs_floor
        dev_subjs_num   = int(devtest_floor / 2)
        test_subjs_num  = int(devtest_floor / 2)
    else: # the split is not even, so use the ceiling value
        train_subjs_num = train_subjs_ceil
        dev_subjs_num   = int(devtest_ceil / 2)
        test_subjs_num  = int(devtest_ceil / 2)
    
    # Partition the randomized subject list based on the number of train, dev, and test subjects.
    train_subjs = subjects_list[0:train_subjs_num]
    dev_subjs   = subjects_list[train_subjs_num:(train_subjs_num + dev_subjs_num)]
    test_subjs  = subjects_list[(train_subjs_num + dev_subjs_num):(train_subjs_num + dev_subjs_num + test_subjs_num)]

    assert len(train_subjs) + len(dev_subjs) + len(test_subjs) == total_subjs, "Number of subjects in each split does not add up to the total number of subjects"
    
    return train_subjs, dev_subjs, test_subjs   