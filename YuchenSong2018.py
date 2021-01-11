#!/usr/bin/python
# %% [markdown]
# # # 4
# # # 
# # GRU for SoC only by Yuchen Song - 2018
# 

# %%
import os

from parser.WindowGenerator import WindowGenerator                       # OS, SYS, argc functions
import pandas as pd             # File read
import matplotlib as mpl        # Plot functionality
import matplotlib.pyplot as plt
import tensorflow as tf         # Tensorflow and Numpy replacement
import tensorflow.experimental.numpy as tnp 
import logging

from sys import platform        # Get type of OS

from parser.DataGenerator import *

#! Temp Fix
import numpy as np
# %%
# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# Configurage logger and print basics
logging.basicConfig(level=logging.CRITICAL,        
    format='%(asctime)s --> %(levelname)s:%(message)s')
logging.warning("Logger enabled")

logging.debug("\n\n"
    f"MatPlotLib version: {mpl.__version__}\n"
    f"Pandas     version: {pd.__version__}\n"
    f"Tensorflow version: {tf.version.VERSION}\n"
    )
logging.debug("\n\n"
    f"Plot figure size set to {mpl.rcParams['figure.figsize']}\n"
    f"Axes grid: {mpl.rcParams['axes.grid']}"
    )
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
    logging.info("GPU found and memory growth enabled") 
    
    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    logging.info("GPU found") 
    logging.debug(f"\nPhysical GPUs: {len(physical_devices)}"
                  f"\nLogical GPUs: {len(logical_devices)}")
#! For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float32')
# %%
dataGenerator = DataGenerator(train_dir='Data/A123_Matt_Set',
                              valid_dir='Data/A123_Matt_Val',
                              test_dir='Data/A123_Matt_Test',
                              columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                                'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                                ],
                              PROFILE_range = 'FUDS')

# training = dataGenerator.train.loc[:, 
#                         ['Current(A)', 'Voltage(V)', 'Temperature (C)_1']]
# %%
window = WindowGenerator(Data=dataGenerator,
                        input_width=1, label_width=1, shift=0,
                        input_columns=['Current(A)', 'Voltage(V)', 'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=230,
                        includeTarget=False, normaliseLabal=False,
                        shuffleTraining=False)
# ds_train, _, _ = window.train
# ds_valid, _, _ = window.valid
_, ds_train, xx_train, yy_train = window.full_train
_, ds_valid, xx_valid, yy_valid = window.full_valid
x_train = np.array(xx_train)
x_valid = np.array(xx_valid)
y_train = np.array(yy_train)
y_valid = np.array(yy_valid)
# x_train : tnp.ndarray = np.array(list(ds_train.map(
#                             lambda x, _: x[0,:,:]
#                             ).as_numpy_iterator()
#                         )[:-1])
# y_train : tnp.ndarray = np.array(list(ds_train.map(
#                             lambda _, y: y[0]
#                             ).as_numpy_iterator()
#                         )[:-1])
# x_valid : tnp.ndarray = np.array(list(ds_valid.map(
#                             lambda x, _: x[0,:,:]
#                             ).as_numpy_iterator()
#                         )[:-1])
# y_valid : tnp.ndarray = np.array(list(ds_valid.map(
#                             lambda _, y: y[0]
#                             ).as_numpy_iterator()
#                         )[:-1])