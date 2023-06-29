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
from keras.layers import concatenate

#%% Custom metric to evaluate how well regression predictions doing compared to how magnitude that the subject was actually accelerating

@tf.function
def norm_mae(y_true, y_pred):
    norm_true_pred = tf.norm((y_pred - y_true), ord='euclidean', axis=[-2,-1])
    print(f"norm_true_pred: {norm_true_pred}")
    norm_true = tf.norm(y_true, ord='euclidean', axis=[-2,-1])
    print(f"norm_true: {norm_true}")
    out = K.mean(norm_true_pred / K.max(K.stack([norm_true, tf.ones_like(norm_true)]), axis=0), axis=-1)
    print(f"K max K stack line: {K.max(K.stack([norm_true, tf.ones_like(norm_true)]), axis=0)}")
    print(f"norm_true_pred / line: {norm_true_pred / K.max(K.stack([norm_true, tf.ones_like(norm_true)]), axis=0)}")
    return out

    
    #return K.mean(norm_true_pred / np.max(norm_true, tf.ones_like(norm_true)), axis=-1)

#%% Build the model.
# # # CONSTRUCT A VANILLA LSTM MODEL # # #
# Code and hyperparameter initial selection adapted from: https://github.com/alcantarar/Recurrent_GRF_Prediction

def build_model(NUM_TIMESTEPS, NUM_INPUT_FEATS, NUM_OUTPUT_FEATS, NUM_OUT_CONT, NUM_OUT_CLASS, NUM_OUT_DIST, MASK_VALUE,
                lstm_nodes = 512, lstm_act = 'tanh', dropout_rate = 0.4, learning_rate = 0.001):

  # Define model layers:
  model_inputs = keras.Input(shape=(NUM_TIMESTEPS, NUM_INPUT_FEATS)) # num_timesteps, num_features
  model_layer = keras.layers.Masking(mask_value=MASK_VALUE, input_shape=(NUM_TIMESTEPS, NUM_INPUT_FEATS))(model_inputs) # Masking layer to ignore padded time steps
  model_layer = keras.layers.Bidirectional(keras.layers.LSTM(lstm_nodes, activation=lstm_act, return_sequences=True), merge_mode='ave')(model_layer) # Vanilla LSTM unit
  model_layer = keras.layers.Dropout(dropout_rate, seed=42)(model_layer)
  
  # Define separate outputs for types of output features (this is in the order of the output feature struct):
  out_continuous = keras.layers.Dense(NUM_OUT_CONT, activation='linear', name='cont')(model_layer) # for continuous outputs, ie. the GRFs and GRMs (mags: unbounded; unit vecs: -1 to 1 since direction)
  out_class      = keras.layers.Dense(NUM_OUT_CLASS, activation='sigmoid', name='class')(model_layer) # for contact classification (0 or 1)
  out_dist       = keras.layers.Dense(NUM_OUT_DIST, activation='linear', name='dist')(model_layer) # for GRF distribution (0-1)
  
  # Combine these
  model_outputs = [out_continuous, out_class, out_dist]
  
  # Create model object and compile the model:
  model_out = keras.Model(inputs=model_inputs, outputs=model_outputs, name='LSTM')
  opt = keras.optimizers.Adam(learning_rate=learning_rate)
  model_out.compile(optimizer=opt,
                    loss={'cont': 'mean_squared_error', 'class': 'binary_crossentropy', 'dist': 'mean_squared_error'},
                    metrics={'cont': [tf.keras.metrics.RootMeanSquaredError(), norm_mae], 'class': tf.keras.metrics.BinaryAccuracy(), 'dist': [tf.keras.metrics.RootMeanSquaredError(), norm_mae]},
                    run_eagerly=True)

  return model_out

#%% Build the model.
# # # CONSTRUCT A VANILLA LSTM MODEL # # #
# Code and hyperparameter initial selection adapted from: https://github.com/alcantarar/Recurrent_GRF_Prediction

def build_model_old(NUM_TIMESTEPS, NUM_INPUT_FEATS, NUM_OUTPUT_FEATS, NUM_OUT_CONT, NUM_OUT_CLASS, NUM_OUT_DIST, MASK_VALUE,
                lstm_nodes = 512, lstm_act = 'tanh', dropout_rate = 0.4, learning_rate = 0.001):

  # Define model layers:
  model_inputs = keras.Input(shape=(NUM_TIMESTEPS, NUM_INPUT_FEATS)) # num_timesteps, num_features
  model_layer = keras.layers.Masking(mask_value=MASK_VALUE, input_shape=(NUM_TIMESTEPS, NUM_INPUT_FEATS))(model_inputs) # Masking layer to ignore padded time steps
  model_layer = keras.layers.Bidirectional(keras.layers.LSTM(lstm_nodes, activation=lstm_act, return_sequences=True), merge_mode='ave')(model_layer) # Vanilla LSTM unit
  model_layer = keras.layers.Dropout(dropout_rate, seed=42)(model_layer)
  
  # Define separate outputs for types of output features (this is in the order of the output feature struct):
  out_continuous = keras.layers.Dense(NUM_OUT_CONT, activation='linear', name='cont')(model_layer) # for continuous outputs, ie. the GRFs and GRMs (mags: unbounded; unit vecs: -1 to 1 since direction)
  out_class      = keras.layers.Dense(NUM_OUT_CLASS, activation='sigmoid', name='class')(model_layer) # for contact classification (0 or 1)
  out_dist       = keras.layers.Dense(NUM_OUT_DIST, activation='linear', name='dist')(model_layer) # for GRF distribution (0-1)
  
  # Combine these
  #model_outputs = tf.concat([out_continuous, out_class, out_dist], axis=1)
  #model_outputs = concatenate([out_continuous, out_class, out_dist])
  model_outputs = [out_continuous, out_class, out_dist]

  # Need to clip the unit vectors and the GRF distribution
  # unit_ixs = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
  # dist_ixs = [22, 23, 24, 25, 26, 27]
  
  # clipped_outputs = []

  # for i, output in enumerate(model_outputs):
  #     if i in unit_ixs: # clip to -1 to 1
  #         clipped_outputs.append(tf.clip_by_value(output, -1, 1))
  #     elif i in dist_ixs: # clip to 0 to 1
  #         clipped_outputs.append(tf.clip_by_value(output, 0, 1))
  #     else:
  #         clipped_outputs.append(output)
  
  # Specify loss and metrics for different outputs
  # losses = []
  # metrics = []
  
  # for i in range(NUM_OUTPUT_FEATS):
  #     if i == 20 or i == 21: # for classification
  #         losses.append('binary_crossentropy')
  #         metrics.append(['binary_accuracy'])
  #     else:
  #         losses.append('mean_squared_error')
  #         metrics.append(['root_mean_squared_error'])
  
  #losses = ['mean_squared_error', 'binary_crossentropy', 'mean_squared_error']
  #metrics = [['root_mean_squared_error'],['binary_accuracy'], ['root_mean_squared_error']]
  
  # Create model object and compile the model:
  model_out = keras.Model(inputs=model_inputs, outputs=model_outputs, name='LSTM')
  opt = keras.optimizers.Adam(learning_rate=learning_rate)
  # model_out.compile(optimizer=opt,
  #                   loss={'cont': 'mean_squared_error', 'class': 'binary_crossentropy', 'dist': 'mean_squared_error'},
  #                   metrics={'cont': 'root_mean_squared_error', 'class': 'binary_accuracy', 'dist': 'root_mean_squared_error'},
  #                   run_eagerly=True)
  model_out.compile(optimizer=opt,
                    loss={'cont': 'mean_squared_error', 'class': 'binary_crossentropy', 'dist': 'mean_squared_error'},
                    metrics={'cont': tf.keras.metrics.RootMeanSquaredError(), 'class': tf.keras.metrics.BinaryAccuracy(), 'dist': tf.keras.metrics.RootMeanSquaredError()},
                    run_eagerly=True)

  return model_out

#%% Train the model.
# Train the model and return fit history. Contains the training loops.

def train_model(model, model_filename,
                X_train, Y_train_cont, Y_train_class, Y_train_dist,
                X_dev, Y_dev_cont, Y_dev_class, Y_dev_dist,
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
        {'cont': Y_train_cont, 'class': Y_train_class, 'dist': Y_train_dist},
        shuffle=shuffle, # shuffle training data
        epochs=epochs,
        validation_data=(X_dev, {'cont': Y_dev_cont, 'class': Y_dev_class, 'dist': Y_dev_dist}),
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

