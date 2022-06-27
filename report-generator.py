#!/usr/bin/python
# %% [markdown]
# # # Convert 10 attempots into report state to use for article writing
# # #
# #
# %%
import os, sys
import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # File read
from itertools import chain
import tensorflow as tf
import tensorflow_addons as tfa

from py_modules.utils import Locate_Best_Epoch
from py_modules.plotting import predicting_plot

# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
plt.rcParams['figure.facecolor'] = 'white'

Data    : str = '/mnt/LibrarySM/SHARED/Data/'
profiles: list = ['DST', 'US06', 'FUDS']
nLayers : int = 3
nNeurons : int = 131
attempts : str = range(1,11)#range(1, 4)
profile : str = 'FUDS'

file_name : str = 'Chemali2017'#'testHyperParams'
model_name: str = 'ModelsUp-1'

# %% [histories]
def avr_attempts(profile, file_name, model_name, nLayers, nNeurons, criteria='mae'):
    length = pd.read_csv(
                    f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                    f'1-{profile}/1-train-logits.csv').shape[0]
    logits = np.empty(shape=(length,1))
    for a in attempts:
        file = (f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                f'{a}-{profile}/history.csv')
        bestEpoch, err  = Locate_Best_Epoch(file, 'mae')
        
        if err < 0.20:
            logits = np.append(logits, pd.read_csv(
                        f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                        f'{a}-{profile}/{bestEpoch}-train-logits.csv').iloc[:, -1:].values,
                        axis=1)
            # plt.plot(logits[:,-1], label=nNames)
            print(f'Best epoch for {nLayers}x({nNeurons})-{a} is: {bestEpoch} with {err}')
                    
        else:
            print(f'XXX--->> Failed model at {nLayers}x({nNeurons})-{a} with {err}')
                    # plt.plot(pd.read_csv(
                    #     f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                    #     f'{a}-{profile}/{bestEpoch}-train-logits.csv').iloc[:, -1].values)
            # plt.figure()
    # plt.plot(logits.mean(axis=1), label=nNames)
    return bestEpoch, logits

file_name : str = 'Chemali2017'#'testHyperParams'
model_name: str = 'ModelsUp-1'
titles = {}
data = {}
profile = 'FUDS'
nLayers = 3
nNeurons= 131
TableRecords = pd.DataFrame(
        columns = ['Layers', 'Neurons', 'minEpochs',   # Information
                    'size', 'tr_time', 'Success_rate', # Propertirs
                    'alpha', 'c',    # Line Fit
                    'Ts_time/sample', # Speed
                    #'BestEpoch', 'Lowest_tr_mae', 'Lowest_vl_mae',
                    'DST_tr_mae(%)', 'US06_tr_mae(%)', 'FUDS_tr_mae(%)',
                    #'DST_vl_mae(%)', 'US06_vl_mae(%)', 'FUDS_vl_mae(%)',
                    #'avr_tr_mae(%)'
                    ]
    )

# %%
# Create average plots
MAE     = tf.metrics.MeanAbsoluteError()
RMSE    = tf.metrics.RootMeanSquaredError()
RSquare = tfa.metrics.RSquare(dtype=tf.float32)
for profile in profiles:
    length = pd.read_csv(
                    f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                    f'1-{profile}/1-train-logits.csv').shape[0]
    logits = np.empty(shape=(length,1))
    if profile == 'FUDS':
        attempts = range(1,10)
    else: 
        attempts = range(1,11)
    for a in attempts:
        file = (f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                f'{a}-{profile}/history.csv')
        bestEpoch, err  = Locate_Best_Epoch(file, 'mae')

        if bestEpoch < 5:
            print(f'Interrupted model at {profile} attempt: {a}')
        else:
            logits = np.append(logits, pd.read_csv(
                    f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                    f'{a}-{profile}/{bestEpoch}-train-logits.csv').iloc[:, -1:].values,
                    axis=1)
    y_data = pd.read_csv(f'{Data}/validation/{profile}_yt_valid.csv').iloc[:,-1]
    # for i in attempts:
    #     print(np.mean(y_data-logits[:,i]))
    y_result = logits[:,1:].mean(axis=1)
    MAE.update_state(y_true=y_data,     y_pred=y_result)
    RMSE.update_state(y_true=y_data,    y_pred=y_result)
    RSquare.update_state(y_true=y_data, y_pred=y_result)
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
            y_data-y_result)))
    predicting_plot(profile=profile, file_name=model_name,
            model_loc=f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/',
            model_type='LSTM average',
            iEpoch=f'10-attempts',
            Y=y_data,
            PRED=y_result,
            RMS=RMS,
            val_perf=[0,
                      MAE.result(),
                      RMSE.result(),
                      RSquare.result()],
            TAIL=y_data.shape[0],
            save_plot=True)
    # plt.plot(y_data)
    # plt.plot(logits[:,1:].mean(axis=1))

# %%