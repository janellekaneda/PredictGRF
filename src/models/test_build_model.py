# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 22:18:01 2023

@author: jkaneda
"""
#%% IMPORTS
from tensorflow import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.layers import concatenate

#%% FUNCTION

def build_model(NUM_TIMESTEPS, NUM_INPUT_FEATS, NUM_OUTPUT_FEATS, NUM_OUT_CONT, NUM_OUT_CLASS, NUM_OUT_DIST, MASK_VALUE, lstm_nodes = 512, lstm_act = 'tanh', dropout_rate = 0.4, learning_rate = 0.001):

  # Define model layers:
  model_inputs = keras.Input(shape=(NUM_TIMESTEPS, NUM_INPUT_FEATS)) # num_timesteps, num_features
  model_layer = keras.layers.Masking(mask_value=MASK_VALUE, input_shape=(NUM_TIMESTEPS, NUM_INPUT_FEATS))(model_inputs) # Masking layer to ignore padded time steps
  print("Input Mask Values:", model_layer._keras_mask)
  model_layer = keras.layers.Bidirectional(keras.layers.LSTM(lstm_nodes, activation=lstm_act, return_sequences=True), merge_mode='ave')(model_layer) # Vanilla LSTM unit
  model_layer = keras.layers.Dropout(dropout_rate, seed=42)(model_layer)
  
  # Define separate outputs for types of output features (this is in the order of the output feature struct):
  out_continuous = keras.layers.Dense(NUM_OUT_CONT, activation='linear', name='cont')(model_layer) # for continuous outputs, ie. the GRFs and GRMs (mags: unbounded; unit vecs: -1 to 1 since direction)
  out_class = keras.layers.Dense(NUM_OUT_CLASS, activation='sigmoid', name='class')(model_layer) # for contact classification (0 or 1)
  out_dist = keras.layers.Dense(NUM_OUT_DIST, activation='linear', name='dist')(model_layer) # for GRF distribution (0-1)
  
  # Combine these
  model_outputs = [out_continuous, out_class, out_dist]
  
  # Create model object and compile the model:
  model_out = keras.Model(inputs=model_inputs, outputs=model_outputs, name='LSTM')
  print("Output Mask Values:", model_out.output_mask)
  opt = keras.optimizers.Adam(learning_rate=learning_rate)
  model_out.compile(optimizer=opt, loss={'cont': 'mean_squared_error', 'class': 'binary_crossentropy', 'dist': 'mean_squared_error'}, metrics={'cont': [tf.keras.metrics.RootMeanSquaredError()], 'class': tf.keras.metrics.BinaryAccuracy(), 'dist': [tf.keras.metrics.RootMeanSquaredError()]}, run_eagerly=True)

  return model_out





#%% TEST

EXP_TYPE = 'with_physicsinputs'

# Hyperparameters:
LSTM_NODES = 448
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.0001

MASK_VALUE = 999.
MAX_TIMESTEPS = 112
if EXP_TYPE == 'with_physicsinputs': NUM_INPUT_FEATS = 36
if EXP_TYPE == 'without_physicsinputs': NUM_INPUT_FEATS = 28
NUM_OUTPUT_FEATS = 28
BATCH_SIZE = 32
EPOCHS = 100

SEED = 43

# Load sample data
sample_input = np.load(r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\processed\normalized\with_physicsinputs\X_train_norm_100Hz_RawDist.npy")
sample_input = sample_input[[0,100],:,:]
print(f"sample input 0: {sample_input[0,:,:]}")
sample_output =  np.load(r"W:\OA_GaitRetraining\Janelle\CS230\ForUpload\data\processed\normalized\with_physicsinputs\Y_train_norm_100Hz_RawDist.npy")
sample_output = sample_output[[0,100],:,:]

model = build_model(MAX_TIMESTEPS, NUM_INPUT_FEATS, NUM_OUTPUT_FEATS, 20, 2, 6, MASK_VALUE, LSTM_NODES, 'tanh', DROPOUT_RATE, LEARNING_RATE)
preds = model.predict(sample_input)
preds_cont = preds[1]
print(f"preds 0: {preds_cont[0,:,:]}")

mask_layer_output = model.get_layer('masking').output
#get_mask_values = K.function([model.input], [mask_layer_output._keras_mask])
#mask_values = get_mask_values([sample_input])[0]
#print("Mask Values:", mask_values)
mask_layer = model.get_layer('masking')
mask_values = mask_layer.compute_mask(sample_input)
print(f"mask values: {mask_values}")
cont_mask = model.get_layer('cont').compute_mask(sample_input)
print(f"cont values: {cont_mask}")
out_mask = model.output_mask
print(f"out mask: {out_mask}")

# Apply mask to predictions
#masked_preds_cont = preds_cont[:,:,19] * tf.cast(mask_values, dtype=tf.float32)

# Print the masked predictions
#print(f"PREDS WITH MASK: {masked_preds_cont[0, :]}")
#print(f"PREDS WITHOUT MASK: {preds_cont[0,:,0]}")

sample_preds = preds_cont[0,:,0]
sample_mask = mask_values[0]
print(f"MASKED PREDS: {sample_preds[sample_mask]}")
print(f"UNMASKED PREDS: {sample_preds}")




fit_history = model.fit(
    sample_input,
    {'cont': sample_output[:,:,0:20], 'class': sample_output[:,:,20:22], 'dist': sample_output[:,:,22:]},
    shuffle=False, # shuffle training data
    epochs=1,
    verbose=1,
    batch_size=1
    )