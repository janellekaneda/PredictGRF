# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 21:18:39 2022

@author: jkaneda

Visualize np.gradient
"""
import matplotlib.pyplot as plt
import numpy as np
import os

BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload' 

dim = 1
Y_dev = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'not_normalized', 'Y_dev.npy'), allow_pickle=True)
Y_dev_grad = np.gradient(Y_dev)[dim]

#%% subset
Y_dev_subtime= Y_dev[:,0:30,:]
Y_dev_subgrad = np.gradient(Y_dev_subtime)[dim]

#%%

plt.figure()
plt.plot(Y_dev_subtime[10,:,1])
plt.figure()
plt.plot(Y_dev_subgrad[10,:,1])

# =============================================================================
# 
# plt.figure()
# plt.plot(Y_dev[0,0:38,1])
# plt.figure()
# plt.plot(Y_dev_grad[0,0:38,1])
# 
# =============================================================================

plt.figure()
plt.plot(Y_dev[0,:,0],label='y_dev')
plt.plot(Y_dev_grad[0,:,0],label='grad')
plt.legend()