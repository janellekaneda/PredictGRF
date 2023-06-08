# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:55:50 2022

@author: jkaneda

Contains functions for hyperparameter tuning.
"""

#%% IMPORTS

from tensorflow import keras
import keras_tuner as kt

#%% Build model for hyperparam tuning

def model_builder_wrapper(MAX_TIMESTEPS, NUM_INPUT_FEATS, NUM_OUTPUT_FEATS, MASK_VALUE, SEED,
                          loss_func, alpha, tune_loss = False,
                          metrics = 'root_mean_squared_error',
                          min_nodes = 32, max_nodes = 512, nodes_step = 32,
                          dropout_values = [0.4, 0.5, 0.6],
                          lr_values = [1e-2, 1e-3, 1e-4],
                          grad_loss_alpha_values = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
    
    def model_builder(hp):
      model = keras.Sequential()
      model.add(keras.Input(shape=(MAX_TIMESTEPS, NUM_INPUT_FEATS))) # input layer
      model.add(keras.layers.Masking(mask_value=MASK_VALUE, input_shape=(MAX_TIMESTEPS, NUM_INPUT_FEATS))) # maskign layer for padded time steps
    
      # Tune number of units in LSTM layer
      # Choose an optimal value between 32-512
      hp_units = hp.Int('units', min_value=min_nodes, max_value=max_nodes, step=nodes_step)
      model.add(keras.layers.Bidirectional(keras.layers.LSTM(hp_units, activation='tanh', return_sequences=True), merge_mode='ave'))
    
      # Tune the dropout rate
      hp_dropout_rate = hp.Choice('dropout_rate', values=dropout_values)
      model.add(keras.layers.Dropout(hp_dropout_rate, seed=SEED))
    
      # Don't tune output layer
      model.add(keras.layers.Dense(NUM_OUTPUT_FEATS, activation='linear'))
    
      # Tune learning rate
      hp_learning_rate = hp.Choice('learning_rate', values=lr_values)
      
      if tune_loss:
          # Tune alpha in the custom loss function
          hp_grad_loss_alpha = hp.Choice('grad_loss_alpha', values=grad_loss_alpha_values)
          loss = loss_func(hp_grad_loss_alpha, 1-hp_grad_loss_alpha)
      else: 
          loss = loss_func(alpha, (1-alpha))
      
      opt = keras.optimizers.Adam(learning_rate=hp_learning_rate)
      model.compile(optimizer=opt,
                    loss=loss,
                    metrics=metrics,
                    run_eagerly=True)
      
      return model

    return model_builder

#%% Run the tuner

def run_tuner(model_builder, objective, 
              directory, project_name,
              X_train, Y_train, X_dev, Y_dev,
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
                 Y_train,
                 shuffle=True,
                 epochs=EPOCHS,
                 validation_data=(X_dev,Y_dev),
                 batch_size=BATCH_SIZE,
                 callbacks=callbacks)
    
    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    return best_hps
