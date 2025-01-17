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
# Get the fanyc learning rate degradation
import re
mods = 'Modds'
attempt = 111
file = (f'{mods}/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                f'{attempt}-{profile}/history.csv')
hists = pd.read_csv(file)
# plt.plot(history['learn_r'])
def get_faulties(dir):
    faulties : list = []
    regex : re.Pattern = re.compile('(.*faulty-history.csv$)')
    for _, _, files in os.walk(dir):
        for file in files:
            if regex.match(file):
                faulties.append(file)
    return faulties

dir = (f'{mods}/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                f'{attempt}-{profile}/')
faulties = get_faulties(dir)

f_histories = []
for f in faulties:
    temp = pd.read_csv(f'{mods}/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                f'{attempt}-{profile}/{f}')
    temp = temp.rename(columns={'learning_rate':'learn_r'})    
    f_histories.append(temp)
# for i in f_histories:
#     hists =pd.concat([hists, i])
# entire_hist = hists.sort(['Epoch'], ascending=True)
# %%
start = 0
l_values = np.array([0.001])
f = 0
for faulty in faulties[:9]:
    end = int(faulties[f][:2])
    l_values = np.concatenate((l_values, hists[start:end]['learn_r'].values))
    l_values = np.concatenate((l_values, f_histories[f]['learn_r'].values))
    start = end
    f += 1
# print(l_values)
# %%
l_values = l_values[:40]
fig, ax = plt.subplots(figsize=(28,12), dpi=600)
fig.suptitle('Learning rate degradation over single training',
              fontsize=40)
ax.plot(l_values, '-o', color='#0000ff')
# ax.set_yticks(l_values[::10])
ax.set_xlabel('Passes', fontsize=32)
ax.tick_params(axis='both', labelsize=36)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()
fig.savefig(f'{mods}/{model_name}/{nLayers}x{file_name}-({nNeurons})/{attempt}-{profile}/l_rate.svg')

# %%
