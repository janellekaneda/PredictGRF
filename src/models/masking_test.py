# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 23:20:11 2023

@author: jkaneda
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define the model
model = keras.Sequential([
    keras.layers.Masking(mask_value=0.0, input_shape=(2, 5)),  # Masking layer
    keras.layers.Bidirectional(keras.layers.LSTM(10, return_sequences=True)),
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Prepare input data with masked values
input_data = np.array([
    [[1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0]],  # First sequence with masked values
    [[6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]],  # Second sequence without masked values
])

# Predict with the model
predictions = model.predict(input_data)

# Extract output from the bidirectional LSTM layer
lstm_output = model.layers[1].output

# Create a Keras function to get the actual values of the LSTM output
get_output_values = keras.backend.function([model.input], [lstm_output])

# Get the values of the LSTM output for the first sequence
output_values = get_output_values([input_data])[0]

# Print the output values for the first sequence
print(output_values[0])  # Assuming batch size is 1


