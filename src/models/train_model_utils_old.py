# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:06:04 2022

@author: jkaneda

These functions create the model and custom loss function for training.
"""
#%% IMPORTS
from tensorflow import keras
import keras.backend as K
import tensorflow as tf
import numpy as np

#%% Build the model.
# # # CONSTRUCT A VANILLA LSTM MODEL # # #
# Code and hyperparameter initial selection adapted from: https://github.com/alcantarar/Recurrent_GRF_Prediction

def build_model(NUM_TIMESTEPS, NUM_INPUT_FEATS, NUM_OUTPUT_FEATS, MASK_VALUE,
                lstm_nodes = 512, lstm_act = 'tanh', dropout_rate = 0.4, dense_act = 'linear', learning_rate = 0.001, loss = 'mean_squared_error'):

  # Define model layers:
  model_inputs = keras.Input(shape=(NUM_TIMESTEPS, NUM_INPUT_FEATS)) # num_timesteps, num_features
  model_layer = keras.layers.Masking(mask_value=MASK_VALUE, input_shape=(NUM_TIMESTEPS, NUM_INPUT_FEATS))(model_inputs) # Masking layer to ignore padded time steps
  model_layer = keras.layers.Bidirectional(keras.layers.LSTM(lstm_nodes, activation=lstm_act, return_sequences=True), merge_mode='ave')(model_layer) # Vanilla LSTM unit
  model_layer = keras.layers.Dropout(dropout_rate, seed=42)(model_layer)
  model_outputs = keras.layers.Dense(NUM_OUTPUT_FEATS, activation=dense_act)(model_layer) # Dense layer to have as many output nodes for ground reaction feats as needed

  # Create model object and compile the model:
  model_out = keras.Model(inputs=model_inputs, outputs=model_outputs, name='LSTM')
  opt = keras.optimizers.Adam(learning_rate=learning_rate)
  model_out.compile(optimizer=opt,
                    loss=loss,
                    run_eagerly=True)

  return model_out

#%% Train the model.
# Train the model and return fit history. Contains the training loops.

def train_model(model, model_filename, X_train, Y_train, X_dev, Y_dev, shuffle = True, epochs = 50, batch_size = 32):

    # Create an early stopping callback.
    earlystopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        verbose=1, 
        patience=10,
        min_delta=0.001, 
        restore_best_weights=True
        )
    
    # Create a model checkpoint to save info at a given frequency.
    checkpoint = keras.callbacks.ModelCheckpoint(
        model_filename,
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True,
        save_weights_only=False
        )
    
    # Train the model and record the training history.
    fit_history = model.fit(
        X_train,
        Y_train,
        shuffle=shuffle, # shuffle training data
        epochs=epochs, # Keep 50 epochs and early stopping
        validation_data=(X_dev, Y_dev),
        verbose=1,
        batch_size=batch_size,
        callbacks=[checkpoint, earlystopping]
        )
    
    return fit_history

#%% Custom loss function to incorporate the mean square error of the gradients of the 
# ground reaction load signals.

def weighted_grad_loss(reg_weight, grad_weight):
    
    def loss_function(y_true, y_pred):
        
        # Extract the data label output feature from y_true.
        dataset_ids = y_true[:,0,-1] # labels should be the same for all time steps for a given trial
        dataset_ids = dataset_ids.numpy() 
        
        # Extract the actual output data (excluding the last label feature).
        y_true = y_true[:,:,:-1]
        y_pred = y_pred[:,:,:-1]

        # Compute the mean square error on y_true and y_pred.
        reg_loss = K.mean(K.square(y_pred - y_true), axis=-1)
        
        # # Compute the mean error of the gradients of y_true and y_pred. # #
        # Compute the gradients.
        y_true_grad = np.gradient(y_true.numpy())[-1] # convert to Numpy to use Numpy's gradient function
        y_pred_grad = np.gradient(y_pred.numpy())[-1] # we want the last element of the returned list, which is the gradients along the last dimension (output feats)
        
        # Convert the gradients to tensors.
        y_true_grad = tf.convert_to_tensor(y_true_grad, dtype = tf.float32)
        y_pred_grad = tf.convert_to_tensor(y_pred_grad, dtype = tf.float32)
        
        # Compute the mean square error of the gradients.
        grad_loss = K.mean(K.square(y_pred_grad - y_true_grad), axis=-1)
        
        # Return the final loss as a weighted sum of the two losses.
        loss = reg_weight * reg_loss + grad_weight * grad_loss
        
        
        # Multiply loss by weight vector based on dataset type.
        weights = np.ones(len(dataset_ids)) * ((6800/231) - 1) #6800 = total num walking examples, 231 = total num examples per task in ACL set, -1 so we can add that back in after multiply with labels
        weights = weights * dataset_ids # zero out weights for OAGR dataset
        weights = weights + 1 # add one so the weight on the loss is 1 for OAGR, and 6800/231 for ACL
        weights = weights[:, np.newaxis]
        weights = tf.convert_to_tensor(weights, dtype = tf.float32)

        loss = loss * weights
        
        return loss
    
    return loss_function

#%% Custom loss function to incorporate the mean square error of the gradients of the 
# ground reaction load signals (without task weights).

# =============================================================================
# def weighted_grad_loss(reg_weight, grad_weight):
#     
#     def loss_function(y_true, y_pred):
#         
#         # Compute the mean square error on y_true and y_pred.
#         reg_loss = K.mean(K.square(y_pred - y_true), axis=-1)
#         
#         # # Compute the mean error of the gradients of y_true and y_pred. # #
#         # Compute the gradients.
#         y_true_grad = np.gradient(y_true.numpy())[-1] # convert to Numpy to use Numpy's gradient function
#         y_pred_grad = np.gradient(y_pred.numpy())[-1] # we want the last element of the returned list, which is the gradients along the last dimension (output feats)
#         
#         # Convert the gradients to tensors.
#         y_true_grad = tf.convert_to_tensor(y_true_grad)
#         y_pred_grad = tf.convert_to_tensor(y_pred_grad)
#         
#         # Compute the mean square error of the gradients.
#         grad_loss = K.mean(K.square(y_pred_grad - y_true_grad), axis=-1)
#         
#         # Return the final loss as a weighted sum of the two losses.
#         loss = reg_weight * reg_loss + grad_weight * grad_loss
#         
#         return loss
#     
#     return loss_function
# =============================================================================

#%% Shuffle the given data and corresponding data labels.

# =============================================================================
# def shuffle_data_and_labels(X_data, Y_data, labels, seed):
#     """
#     Shuffle the given X and Y data and the corresponding data labels, so that
#     they are shuffled in the same way.
#     
#     Adapted from: https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
# 
#     """
#     # Check that the inputs all have the same number of examples.
#     assert X_data.shape[0] == Y_data.shape[0] == len(labels)
#     
#     # Get number of examples.
#     num_ex = len(labels)
#     
#     # Get the permutation indices.
#     perm_ix = np.random.RandomState(seed=seed).permutation(num_ex)
#     
#     return X_data[perm_ix, :, :], Y_data[perm_ix, :, :], labels[perm_ix]
# =============================================================================

