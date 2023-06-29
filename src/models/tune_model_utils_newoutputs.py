# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:55:50 2022

@author: jkaneda

Contains functions for hyperparameter tuning.
"""
#%% IMPORTS
from tensorflow import keras
import keras_tuner as kt
import tensorflow as tf
import keras.backend as K

#%% Custom metric to evaluate how well regression predictions doing compared to how magnitude that the subject was actually accelerating

def norm_mae(y_true, y_pred):
    norm_true_pred = keras.utils.normalize((y_pred - y_true), axis=1, order=1)
    #print(f"norm_true_pred: {norm_true_pred}")
    norm_true = keras.utils.normalize(y_true, axis=1, order=1)
    #print(f"norm_true: {norm_true}")
    return K.mean(norm_true_pred / K.max(K.stack([norm_true, tf.ones_like(norm_true)]), axis=0), axis=-1)
    
    #return K.mean(norm_true_pred / np.max(norm_true, tf.ones_like(norm_true)), axis=-1)

#%% Build model for hyperparam tuning

def model_builder_wrapper(NUM_TIMESTEPS, NUM_INPUT_FEATS, NUM_OUTPUT_FEATS, NUM_OUT_CONT, NUM_OUT_CLASS, NUM_OUT_DIST,
                          MASK_VALUE, SEED,
                          lstm_act = 'tanh',
                          min_nodes = 32, max_nodes = 512, nodes_step = 32,
                          dropout_values = [0.5, 0.6, 0.7, 0.8, 0.9],
                          lr_values = [1e-2, 1e-3, 1e-4]):
    
    def model_builder(hp):
        
        # Define model layers:
        model_inputs = keras.Input(shape=(NUM_TIMESTEPS, NUM_INPUT_FEATS)) # num_timesteps, num_features
        model_layer = keras.layers.Masking(mask_value=MASK_VALUE, input_shape=(NUM_TIMESTEPS, NUM_INPUT_FEATS))(model_inputs) # Masking layer to ignore padded time steps
        
        # Tune number of units in LSTM layer
        hp_units = hp.Int('units', min_value=min_nodes, max_value=max_nodes, step=nodes_step)
        model_layer = keras.layers.Bidirectional(keras.layers.LSTM(hp_units, activation=lstm_act, return_sequences=True), merge_mode='ave')(model_layer) # Vanilla LSTM unit
        
        # Tune the dropout rate
        hp_dropout_rate = hp.Choice('dropout_rate', values=dropout_values)
        model_layer = keras.layers.Dropout(hp_dropout_rate, seed=SEED)(model_layer)
        
        # Define separate outputs for types of output features (this is in the order of the output feature struct):
        out_continuous = keras.layers.Dense(NUM_OUT_CONT, activation='linear', name='cont')(model_layer) # for continuous outputs, ie. the GRFs and GRMs (mags: unbounded; unit vecs: -1 to 1 since direction)
        out_class      = keras.layers.Dense(NUM_OUT_CLASS, activation='sigmoid', name='class')(model_layer) # for contact classification (0 or 1)
        out_dist       = keras.layers.Dense(NUM_OUT_DIST, activation='linear', name='dist')(model_layer) # for GRF distribution (0-1)
        
        # Combine these
        model_outputs = [out_continuous, out_class, out_dist]
        
        # Create model object and compile the model:
        model_out = keras.Model(inputs=model_inputs, outputs=model_outputs, name='LSTM')
        # Tune learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=lr_values)
        opt = keras.optimizers.Adam(learning_rate=hp_learning_rate)
        model_out.compile(optimizer=opt,
                          loss={'cont': 'mean_squared_error', 'class': 'binary_crossentropy', 'dist': 'mean_squared_error'},
                          metrics={'cont': [tf.keras.metrics.RootMeanSquaredError(), norm_mae], 'class': tf.keras.metrics.BinaryAccuracy(), 'dist': [tf.keras.metrics.RootMeanSquaredError(), norm_mae]},
                          run_eagerly=True)
        
        return model_out

    return model_builder

#%% Run the tuner

def run_tuner(model_builder, objective, 
              directory, project_name,
              X_train, Y_train_cont, Y_train_class, Y_train_dist,
              X_dev, Y_dev_cont, Y_dev_class, Y_dev_dist,
              EPOCHS, BATCH_SIZE,
              callbacks):
    
    tuner = kt.Hyperband(model_builder,
                     objective=objective,
                     max_epochs=EPOCHS,
                     factor=3,
                     directory=directory,
                     project_name=project_name)
    
    
    # Run the search!
    tuner.search(X_train,
                 {'cont': Y_train_cont, 'class': Y_train_class, 'dist': Y_train_dist},
                 shuffle=True,
                 epochs=EPOCHS,
                 validation_data=(X_dev, {'cont': Y_dev_cont, 'class': Y_dev_class, 'dist': Y_dev_dist}),
                 batch_size=BATCH_SIZE,
                 callbacks=callbacks)
    
    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    return best_hps
