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
                lstm_nodes = 512, lstm_act = 'tanh', dropout_rate = 0.4, dense_act = 'linear', learning_rate = 0.001, loss = 'mean_squared_error', metrics=['root_mean_squared_error']):

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
                    metrics=metrics,
                    run_eagerly=True)

  return model_out

#%% Train the model.
# Train the model and return fit history. Contains the training loops.

def train_model(model, model_filename, X_train, Y_train, X_dev, Y_dev, use_earlystopping = False, shuffle = True, epochs = 50, batch_size = 32):

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
        Y_train,
        shuffle=shuffle, # shuffle training data
        epochs=epochs,
        validation_data=(X_dev, Y_dev),
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

