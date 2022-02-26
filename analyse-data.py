#!/usr/bin/python
# %% [markdown]
# # # Analyze the results were collected
# # #
# #
# %%
import os, sys
import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # File read

# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
# %%
Data    : str = 'Data/'
profiles: list = ['DST', 'US06', 'FUDS']
neurons : list = [ 32, 65, 131, 262, 524 ]
layers : range = range(1, 5)
attempt : str = '1'
profile : str = 'DST'

file_name : str = 'Chemali2017'
model_name: str = 'Model №1'

# %%
names     : list = []
histories : list = []
profile : str = 'FUDS'
for nLayers in layers:
    nNames = []
    nHistories = []
    for nNeurons in neurons:
        nNames.append(f'{nLayers}x({nNeurons})-{attempt}-{profile}')
        nHistories.append(
            pd.read_csv(f'Mods/{nLayers}x{file_name}-({nNeurons})/'
                        f'{attempt}-{profile}/history.csv')
            )
    names.append(nNames)
    histories.append(nHistories)
nNames = []
nHistories = []
# %%
#? MAE
for i in range(len(neurons)):
    # plt.figure()
    plt.plot(histories[0][i].iloc[:, 1], label=names[0][i])
plt.legend()
plt.title('Model MAE')
plt.ylim([0.0, 0.05])
# %%
#? Train_MAE
for i in range(len(neurons)):
    # plt.figure()
    plt.plot(histories[0][i].iloc[:, 6], label=names[0][i])
plt.legend()
plt.title('Training MAE')
plt.ylim([0.0, 0.1])
# %%
#? Evaluate_MAE
for i in range(len(neurons)):
    # plt.figure()
    plt.plot(histories[0][i].iloc[:, 10], label=names[0][i])
plt.legend()
plt.title('Evaluate MAE')
plt.ylim(0.0, 0.15)
# %%
# %%
#! Bar plots for weight and timing