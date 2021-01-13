#!/usr/bin/python
# %% [markdown]
# # # 2
# # # 
# # GRU for SoC 
# 

# %%
import os                       # OS, SYS, argc functions

from parser.WindowGenerator import WindowGenerator                       
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
GPU=0
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
TestCase : int = 2
print("Performint Test Case Number: {} with Devive: {} \n".format(TestCase, GPU))
if TestCase == 1:
    window = WindowGenerator(Data=dataGenerator,
                            input_width=500, label_width=1, shift=1,
                            input_columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1'
                                ],
                            label_columns=['SoC(%)'], batch=1,
                            includeTarget=True, normaliseLabal=False,
                            shuffleTraining=False)
    _, ds_train, xx_train, yy_train = window.full_train
    _, ds_valid, xx_valid, yy_valid = window.full_valid
    x_train = np.array(xx_train[:,:,3:])
    x_valid = np.array(xx_valid[:,:,3:])
    y_train = np.array(yy_train)
    y_valid = np.array(yy_valid)


    gru_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.InputLayer(input_shape=x_train.shape[-2:],batch_size=None),
        tf.keras.layers.GRU(
            units=480, activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.2,
            recurrent_dropout=0.0, return_sequences=False, return_state=False,
            go_backwards=False, stateful=False, unroll=False, time_major=False,
            reset_after=True
        ),
        #tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=12, activation='sigmoid'),
        tf.keras.layers.Dense(units=1,
                              activation=None)
    ])
elif TestCase == 2:
    window = WindowGenerator(Data=dataGenerator,
                            input_width=1, label_width=1, shift=1,
                            input_columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1'
                                ],
                            label_columns=['SoC(%)'], batch=500,
                            includeTarget=True, normaliseLabal=False,
                            shuffleTraining=False)
    _, ds_train, xx_train, yy_train = window.full_train
    _, ds_valid, xx_valid, yy_valid = window.full_valid
    x_train = np.array(xx_train[:,:,3:])
    x_valid = np.array(xx_valid[:,:,3:])
    y_train = np.array(yy_train)
    y_valid = np.array(yy_valid)
    
    
    gru_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape=(1, 500, 1)),
        tf.keras.layers.GRU(
            units=480, activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.2,
            recurrent_dropout=0.0, return_sequences=False, return_state=False,
            go_backwards=False, stateful=True, unroll=False, time_major=False,
            reset_after=True
        ),
        #tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
        tf.keras.layers.Dense(units=12, activation='sigmoid'),
        tf.keras.layers.Dense(units=1,
                              activation=None)
    ])

elif TestCase == 3:
    window = WindowGenerator(Data=dataGenerator,
                            input_width=1, label_width=1, shift=1,
                            input_columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1'
                                ],
                            label_columns=['SoC(%)'], batch=500,
                            includeTarget=True, normaliseLabal=False,
                            shuffleTraining=False)
    _, ds_train, xx_train, yy_train = window.full_train
    _, ds_valid, xx_valid, yy_valid = window.full_valid
    x_train = np.array(xx_train[:,:,3:])
    x_valid = np.array(xx_valid[:,:,3:])
    y_train = np.array(yy_train)
    y_valid = np.array(yy_valid)

    gru_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape=(1, 500, 1)),
        tf.keras.layers.GRU(
            units=480, activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.2,
            recurrent_dropout=0.0, return_sequences=False, return_state=False,
            go_backwards=False, stateful=True, unroll=False, time_major=False,
            reset_after=True
        ),
        #tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
        tf.keras.layers.Dense(units=12, activation='sigmoid'),
        tf.keras.layers.Dense(units=1,
                              activation=None)
    ])


gru_model.compile(loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(learning_rate=10e-04,
                    beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Adam'),
            metrics=[tf.metrics.MeanAbsoluteError(),
                     tf.metrics.RootMeanSquaredError()],
        )

if TestCase == 1:
    history = gru_model.fit(x=x_train, y=y_train, epochs=10,
                        validation_data=(x_valid, y_valid),
                        callbacks=None, batch_size=1, shuffle=True
                        )
    hist_df = pd.DataFrame(history.history)    
if TestCase == 2:
    history = gru_model.fit(x=x_train, y=y_train, epochs=10,
                        validation_data=(x_valid, y_valid),
                        callbacks=None, batch_size=1, shuffle=False
                        )
    hist_df = pd.DataFrame(history.history)
if TestCase == 3:
    mEpoch : int = 10
    iEpoch : int = 1
    while iEpoch < mEpoch-1:
        history = gru_model.fit(x=x_train, y=y_train, epochs=1,
                            validation_data=(x_valid, y_valid),
                            callbacks=None, batch_size=1, shuffle=False
                            )
        gru_model.reset_states()
        iEpoch+=1
    hist_df = pd.DataFrame(history.history)

gru_model.save(f'Models/Experement/TestCase-{TestCase}')
with open(f'Models/Experement/TestCase-{TestCase}.csv', mode='a') as f:
    hist_df.to_csv(f, index=False, header=True)
# %% 

