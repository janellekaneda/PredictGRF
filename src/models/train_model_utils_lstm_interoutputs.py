# -*- coding: utf-8 -*-
"""
Created on Wed July 5 2023

@author: jkaneda

These functions create the model and custom loss function for training. Incorporates intermediate predictions to inform final GRF prediction.
"""
#%% IMPORTS
from tensorflow import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.layers import concatenate

#%% Custom metric to evaluate how well regression predictions doing compared to how magnitude that the subject was actually accelerating

@tf.function
def norm_mae(y_true, y_pred):
    norm_true_pred = tf.norm((y_pred - y_true), ord='euclidean', axis=[-2,-1])
    #print(f"norm_true_pred: {norm_true_pred}")
    norm_true = tf.norm(y_true, ord='euclidean', axis=[-2,-1])
    #print(f"norm_true: {norm_true}")
    out = K.mean(norm_true_pred / K.max(K.stack([norm_true, tf.ones_like(norm_true)]), axis=0), axis=-1)
    #print(f"K max K stack line: {K.max(K.stack([norm_true, tf.ones_like(norm_true)]), axis=0)}")
    #print(f"norm_true_pred / line: {norm_true_pred / K.max(K.stack([norm_true, tf.ones_like(norm_true)]), axis=0)}")
    return out
    #return K.mean(norm_true_pred / np.max(norm_true, tf.ones_like(norm_true)), axis=-1)

#%% Build the model.
# # # CONSTRUCT A VANILLA LSTM MODEL # # #
# Code and hyperparameter initial selection adapted from: https://github.com/alcantarar/Recurrent_GRF_Prediction

def build_model(NUM_TIMESTEPS, NUM_INPUT_FEATS, MASK_VALUE, SEED,
                lstm_nodes = 512, lstm_act = 'tanh', dropout_rate = 0.4, learning_rate = 0.001):

  # Define initial model layers:
  model_inputs = keras.Input(shape=(NUM_TIMESTEPS, NUM_INPUT_FEATS)) # num_timesteps, num_features
  model_layer = keras.layers.Masking(mask_value=MASK_VALUE, input_shape=(NUM_TIMESTEPS, NUM_INPUT_FEATS))(model_inputs) # Masking layer to ignore padded time steps
  model_layer = keras.layers.Bidirectional(keras.layers.LSTM(lstm_nodes, activation=lstm_act, return_sequences=True), merge_mode='ave')(model_layer) # Vanilla LSTM unit
  model_layer = keras.layers.Dropout(dropout_rate, seed=SEED)(model_layer)
  
  # Define intermediate outputs:
  out_total_grf    = keras.layers.Dense(3, activation='linear', name='tot_grf')(model_layer) # total GRF (normalized by BW) in x, y, z directions (unbounded).
  out_class_r_init = keras.layers.Dense(1, activation='sigmoid', name='class_r')(model_layer) # for contact classification (0 or 1) for right leg.
  out_class_l_init = keras.layers.Dense(1, activation='sigmoid', name='class_l')(model_layer) # for contact classification (0 or 1) for left leg.
  #out_dist_r_init  = keras.layers.Dense(3, activation='linear', name='dist_r_init')(model_layer) # for GRF distribution (0-1), only for right leg. Ratio for left leg is complement.
  out_dist_r       = keras.layers.Dense(3, activation='sigmoid', name='dist_r')(model_layer) # for GRF distribution (0-1), only for right leg. Ratio for left leg is complement.
  
  # Clip out_dist_r so between 0 and 1
  #out_dist_r = keras.layers.Lambda(lambda x: tf.clip_by_value(x, clip_value_min=0, clip_value_max=1), name='dist_r')(out_dist_r_init) # I think this was causing errors

  # # Use information from intermediate outputs to make final output predictions: right and left GRF in x, y, z directions # #
  
  # Get the R and L distribution intermediate predictions:
  out_dist_l = keras.layers.Lambda(lambda x: 1 - x, name='dist_l')(out_dist_r)
  
  # Distribute total GRF to right and left legs:
  tot_grf_r = keras.layers.Multiply()([out_total_grf, out_dist_r])
  tot_grf_l = keras.layers.Multiply()([out_total_grf, out_dist_l])
  
  # Mask based on contact classification predictions:
  out_class_r = keras.layers.Lambda(lambda x: keras.backend.cast(keras.backend.greater(x, 0.5), dtype='float32'))(out_class_r_init) # threshold to binary 0 and 1
  #out_class_r = keras.layers.Reshape((NUM_TIMESTEPS,))(out_class_r)  # Reshape to remove the last dimension
  #out_class_r = keras.layers.RepeatVector(3)(out_class_r) # to mask in each x y z direction
  #out_class_r = keras.layers.Reshape((NUM_TIMESTEPS, 3))(out_class_r) # to get into correct shape for multiplying
  tot_grf_r = keras.layers.Multiply(name='tot_grf_r')([tot_grf_r, out_class_r])

  out_class_l = keras.layers.Lambda(lambda x: keras.backend.cast(keras.backend.greater(x, 0.5), dtype='float32'))(out_class_l_init) # threshold to binary 0 and 1
  #out_class_l = keras.layers.Reshape((NUM_TIMESTEPS,))(out_class_l_init)  # Reshape to remove the last dimension
  #out_class_l = keras.layers.RepeatVector(3)(out_class_l) # to mask in each x y z direction
  #out_class_l = keras.layers.Reshape((NUM_TIMESTEPS, 3))(out_class_l) # to get into correct shape for multiplying
  tot_grf_l = keras.layers.Multiply(name='tot_grf_l')([tot_grf_l, out_class_l])

  # Create model object and compile the model:
  model_out = keras.Model(inputs=model_inputs, outputs=[out_total_grf, out_class_r_init, out_class_l_init, out_dist_r, tot_grf_r, tot_grf_l], name='LSTM')
  opt = keras.optimizers.Adam(learning_rate=learning_rate)
  model_out.compile(optimizer=opt,
                    loss={'tot_grf': 'mean_squared_error', 
                          'class_r': 'binary_crossentropy', 
                          'class_l': 'binary_crossentropy', 
                          'dist_r': 'mean_squared_error', 
                          'tot_grf_r': 'mean_squared_error', 
                          'tot_grf_l': 'mean_squared_error'},
                    metrics={'tot_grf': tf.keras.metrics.RootMeanSquaredError(), 
                             'class_r': tf.keras.metrics.BinaryAccuracy(), 
                             'class_l': tf.keras.metrics.BinaryAccuracy(),
                             'dist_r': tf.keras.metrics.RootMeanSquaredError(), 
                             'tot_grf_r': tf.keras.metrics.RootMeanSquaredError(), 
                             'tot_grf_l': tf.keras.metrics.RootMeanSquaredError()},
                    run_eagerly=True)

  return model_out

#%% Train the model.
# Train the model and return fit history. Contains the training loops.

def train_model(model, model_filename,
                X_train, Y_train_total_grf, Y_train_class_r, Y_train_class_l, Y_train_dist_r, Y_train_grf_r, Y_train_grf_l,
                X_dev, Y_dev_total_grf, Y_dev_class_r, Y_dev_class_l, Y_dev_dist_r, Y_dev_grf_r, Y_dev_grf_l,
                use_earlystopping = False, shuffle = True, epochs = 50, batch_size = 32):

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
    
    if use_earlystopping:
        callbacks_to_use = [checkpoint, earlystopping]
    else:
        callbacks_to_use = [checkpoint]
    
    # Train the model and record the training history.
    fit_history = model.fit(
        X_train,
        {'tot_grf': Y_train_total_grf, 'class_r': Y_train_class_r, 'class_l': Y_train_class_l, 'dist_r': Y_train_dist_r, 'tot_grf_r': Y_train_grf_r, 'tot_grf_l': Y_train_grf_l},
        shuffle=shuffle, # shuffle training data
        epochs=epochs,
        validation_data=(X_dev, {'tot_grf': Y_dev_total_grf, 'class_r': Y_dev_class_r, 'class_l': Y_dev_class_l, 'dist_r': Y_dev_dist_r, 'tot_grf_r': Y_dev_grf_r, 'tot_grf_l': Y_dev_grf_l}),
        verbose=1,
        batch_size=batch_size,
        callbacks=callbacks_to_use
        )
    
    return fit_history

#%% Custom loss function to incorporate the mean square error of the gradients of the 
# ground reaction load signals.

def weighted_grad_loss(reg_weight, grad_weight):
    
    def loss_function(y_true, y_pred):

        # Compute the mean square error on y_true and y_pred.
        reg_loss = K.mean(K.square(y_pred - y_true), axis=-1)
        #reg_loss = K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
        
        # # Compute the mean error of the gradients of y_true and y_pred. # #
        # Compute the gradient of the outputs with respect to timesteps, axis=1.
        y_true_grad = np.gradient(y_true.numpy(), axis=1) # convert to Numpy to use Numpy's gradient function
        y_pred_grad = np.gradient(y_pred.numpy(), axis=1)
        
        # Unit normalize the grad terms.
        y_true_grad = keras.utils.normalize(y_true_grad, axis=1, order=1)
        y_pred_grad = keras.utils.normalize(y_pred_grad, axis=1, order=1)
        
        # Convert the gradients to tensors.
        y_true_grad = tf.convert_to_tensor(y_true_grad, dtype = tf.float32)
        y_pred_grad = tf.convert_to_tensor(y_pred_grad, dtype = tf.float32)
        
        # Compute the mean square error of the gradients.
        grad_loss = K.mean(K.square(y_pred_grad - y_true_grad), axis=-1)
        #grad_loss = K.sqrt(K.mean(K.square(y_pred_grad - y_true_grad), axis=-1))
        
        # Return the final loss as a weighted sum of the two losses.
        # Weights should sum to 1, but code allows for sum(weights) != 1.
        loss = reg_weight * reg_loss + grad_weight * grad_loss
        
        return loss
    
    return loss_function