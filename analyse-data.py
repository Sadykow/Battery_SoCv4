#!/usr/bin/python
# %% [markdown]
# # # Analyze the results were collected
# # #
# #
# %%
from cProfile import label
import os, sys
import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # File read
from itertools import chain

from py_modules.utils import Locate_Best_Epoch

# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
plt.rcParams['figure.facecolor'] = 'white'
# %%
Data    : str = 'Data/'
profiles: list = ['DST', 'US06', 'FUDS']
neurons : list = [ 16, 32, 65, 131, 262, 524 ]
layers : range = range(1, 5)
attempt : str = '1'
profile : str = 'FUDS'

file_name : str = 'Chemali2017'
model_name: str = 'Model №1'

# %%
names     : list = []
histories : list = []
profile : str = 'FUDS'
metric : str = 'mae'
for nLayers in layers:
    nNames = []
    nHistories = []
    for nNeurons in neurons:
        nNames.append(f'{nLayers}x({nNeurons})-{attempt}')
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
def non_zero_min_idx(values : np.array) -> tuple[np.float32, int]:
    """ Find non zero min value and index

    Args:
        values (np.array): array of values

    Returns:
        tuple[np.float32, int]: Tuple with index and value
    """
    min_v, idx_v = float("inf"), sys.maxsize
    for i, v in enumerate(values):
        if(v > 0 and v < min_v):
            min_v = v
            idx_v = i
    return (idx_v, min_v)

def plot_bar(neurons, profile, names, histories, metric, limits):
    fig, axes = plt.subplots(3,2, figsize=(14,12), dpi=600)
    axes[2][1].set_visible(False)
    # axes[2][0].set_position([0.24,0.125,0.228,0.343])
    fig.suptitle(profile)
    minimals : list = []
    indexes : list = []
    for l, ax in enumerate(fig.axes[:-2]):
        for n in range(len(neurons)):
        # try:
            ax.plot(histories[l][n].loc[:, f'{metric}']*100, label=names[l][n])
        # except KeyError:
        #     print(f'Failed at {names[l][n]}')
            minimals.append(
                # [
                # histories[l][n].loc[:, f'{metric}'].idxmin(),
                histories[l][n].loc[:, f'{metric}'].min()
                # ]
            )
            indexes.append(
                histories[l][n].loc[:, f'{metric}'].idxmin()
            )
        minimals.append(0)
        indexes.append(None)
        #! Locate the lowest among them
        ax.set_title(f'{l+1}-Layer')
        ax.set_xlabel("Iterations/Epochs")
        ax.set_ylabel("Error")
        ax.set_ylim(limits)
        ax.legend()
    # fig.show()
    # labels = list(chain.from_iterable(names))
    labels = []
    for l in names:
        for n in l:
            labels.append(n)
        labels.append('')
    y_pos = np.arange(len(labels))
    
    _, ax2 = plt.subplots(figsize=(14,12), dpi=600)
    # hbars = axes[2][0].barh(y_pos, minimals, xerr=indexes, align='center')
    # [v*100 if v > 0 else None for v in [0, 1 ,2 ,3]]
    values = [v*100 for v in minimals]
    hbars = ax2.barh(y_pos, values, align='center')
    ax2.set_yticks(y_pos, labels=labels)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('Error(%)')
    ax2.set_title('Minimal Epoch')

    # Label with specially formatted floats
    ax2.bar_label(hbars, fmt='%.2f')
    for i in range(len(y_pos)):
        if(minimals[i] > 0):
            ax2.text(s=indexes[i], x=0.1, y=y_pos[i], verticalalignment="center", color='w')
    # ax2.set_xlim(right=15)  # adjust xlim to fit labels
    idx, value = non_zero_min_idx(values)
    print(f'The minimal set is {labels[idx]} with error: {value}%')
    #! TODO: Make gaps in the between layers
    print('TODO: Make gaps between layer')

plot_bar(neurons, profile, names, histories, 'mae', [0, 5])
# plot_bar(neurons, profile, names, histories, 'train_mae', [0, 5])
# plot_bar(neurons, profile, names, histories, 'val_mae', [0, 10])
# plot_bar(neurons, profile, names, histories, 'tes_mae', [0, 10])

# %%
bars1 = histories[0][0]['mae']
bars2 = histories[0][0]['train_mae']
barWidth1 = 0.065
barWidth2 = 0.032
x_range = np.arange(len(bars1) / 8, step=0.125)

# r2 = [x + barWidth for x in r1]
# fig, ax = plt.subplots(figsize=(14,12), dpi=600)
plt.bar(x_range, bars1, color='#dce6f2', width=barWidth1/2, edgecolor='#c3d5e8', label='Target')
plt.bar(x_range, bars2, color='#ffc001', width=barWidth2/2, edgecolor='#c3d5e8', label='Actual Value')
# for i, bar in enumerate(bars2):
#     plt.text(i / 8 - 0.015, bar + 1, bar, fontsize=14)
# plt.xticks(x_range, xticks)
plt.tick_params(
    bottom=False,
    left=False,
    labelsize=15
)
plt.rcParams['figure.figsize'] = [25, 7]
plt.axhline(y=0, color='gray')
plt.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.25, -0.3, 0.5, 0.5), prop={'size':20})
plt.box(False)
plt.savefig('plt', bbox_inches = "tight")
plt.show()

# %%
# df = pd.read_csv(f'Mods/Model №1/2xChemali2017-(262)/22-FUDS/history.csv'
#                         )
# df
# # %%
# fig, axs = plt.subplots(1,2)
# axs[0].plot(df['mae'])
# axs[0].set_ylim([0,0.09])
# axs[0].set_title('7-cycles During training')
# axs[0].set_ylabel('Error')
# axs[0].set_xlabel('Iterations/Epochs')
# axs[1].plot(df['val_mae'])
# axs[1].set_title('7-cycles After training')
# axs[1].set_ylim([0,0.09])
# axs[1].set_xlabel('Iterations/Epochs')
# fig.savefig('Mods/Model №1/2xChemali2017-(262)/2-FUDS/test.svg')
# %%
names     : list = []
histories : list = []
logits    : list = []
logits_val: list = []
profile : str = 'FUDS'
metric : str = 'mae'
nLayers : str = '2'
file_name : str = 'Chemali2017'
nNeurons : str = '262'
for attempt in range(1, 9):
    hist_path = f'Mods/Model-№1-1/{nLayers}x{file_name}-({nNeurons})/' \
                f'{attempt}-{profile}/history.csv'
    
    histories.append(
            pd.read_csv(hist_path)
            )
    
    iEpoch, prev_error  = Locate_Best_Epoch(hist_path, 'train_mae')
    
    logits.append(
            pd.read_csv(f'Mods/Model-№1-1/{nLayers}x{file_name}-({nNeurons})/' \
                f'{attempt}-{profile}/{iEpoch}-train-logits.csv')
        )
    iEpoch, prev_error  = Locate_Best_Epoch(hist_path, 'val_mae')
    logits_val.append(
            pd.read_csv(f'Mods/Model-№1-1/{nLayers}x{file_name}-({nNeurons})/' \
                f'{attempt}-{profile}/{iEpoch}-valid-logits.csv')
        )
    print(iEpoch)

# %%
criteria : str = 'val_mae'
avg = pd.DataFrame(data={ '0' : histories[0][criteria]} )
print(histories[0].shape)
plt.plot(histories[0][criteria], label='1-attempt')
for i in range(1, len(histories)):
    print(histories[i].shape)
    plt.plot(histories[i][criteria], label=f'{i+1}-attempt')
    avg[f'{i}'] = histories[i][criteria]
# print(avg)
# avg.plot(subplots=True)
plt.legend()
plt.show()
plt.plot(np.mean(avg, axis=1), linewidth=5, label='mean')

# for nLayers in layers:
#     nNames = []
#     nHistories = []
#     for nNeurons in neurons:
#         nNames.append(f'{nLayers}x({nNeurons})-{attempt}')
#         nHistories.append(
#             pd.read_csv(f'Mods/{nLayers}x{file_name}-({nNeurons})/'
#                         f'{attempt}-{profile}/history.csv')
#             )
#     names.append(nNames)
#     histories.append(nHistories)
# nNames = []
# nHistories = []
# %%
#! Check OS to change SymLink usage
from extractor.DataGenerator import *
Data    : str = 'Data/'
dataGenerator = DataGenerator(train_dir=f'{Data}A123_Matt_Set',
                              valid_dir=f'{Data}A123_Matt_Val',
                              test_dir=f'{Data}A123_Matt_Test',
                              columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                                'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                                ],
                              PROFILE_range = profile)

# %%
avg_logits = pd.DataFrame(data={ '0' : logits[0]['0']} )
for i in range(1, len(logits)):
    avg_logits[f'{i}'] = logits[i]['0']
    plt.plot(avg_logits[f'{i}'])
# print(avg)
# avg.plot(subplots=True)
plt.title('Best performance')
plt.plot(dataGenerator.valid_SoC)
# plt.savefig('all_logits2.png')
plt.show()

plt.figure()
plt.title('Best performance')
plt.plot(dataGenerator.valid_SoC)
plt.plot(np.mean(avg_logits, axis=1), linewidth=5)
# plt.savefig('mean_logits2.png')
# %%
avg_logits = pd.DataFrame(data={ '0' : logits_val[0]['0']} )
for i in range(1, len(logits_val)):
    avg_logits[f'{i}'] = logits_val[i]['0']
    plt.plot(avg_logits[f'{i}'])
plt.title('Best performance')
plt.plot(dataGenerator.valid_SoC)
# plt.savefig('all_logits2.png')
plt.show()

plt.figure()
plt.title('Best performance')
plt.plot(dataGenerator.valid_SoC)
plt.plot(np.mean(avg_logits, axis=1), linewidth=5)
# plt.savefig('mean_logits2.png')
# %%