#!/usr/bin/python
# %% [markdown]
# # Stateful LSTM example based on Cosine wave.
# Design will use onlt SoC to predict RUL. This method will also investigate
#batching mechanism for individual files to speed the process.

# Original file: /cosine_Stateful
# %%
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from parser.WindowGenerator import WindowGenerator
from parser.DataGenerator import *

# Configurate GPUs
# Define plot sizes
#! Select GPU for usage. CPU versions ignores it
GPU=1
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    #! With /device/GPU:1 the output was faster.
    #! need to research more why.
    tf.config.experimental.set_visible_devices(
                            physical_devices[GPU], 'GPU')

    #if GPU == 1:
    tf.config.experimental.set_memory_growth(
                            physical_devices[GPU], True)
    print("GPU found and memory growth enabled") 
    
    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    print("GPU found") 
    print(f"\nPhysical GPUs: {len(physical_devices)}"
          f"\nLogical GPUs: {len(logical_devices)}")
#! For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float32')
# %%
dataGenerator = DataGenerator(train_dir='Data/A123_Matt_Set',
                              valid_dir='Data/A123_Matt_Val',
                              test_dir='Data/A123_Matt_Test',
                              columns=[
                                 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                                ],
                              PROFILE_range = 'FUDS')