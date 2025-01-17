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
from itertools import chain
import tensorflow as tf

from py_modules.utils import Locate_Best_Epoch


# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
plt.rcParams['figure.facecolor'] = 'white'
# %%
Data    : str = 'Data/'
profiles: list = ['FUDS']#['DST', 'US06', 'FUDS']
neurons : list = [131]#[ 131, 262, 524 ]
layers : range = [3]#range(1, 4)
attempts : str = range(1,11)#range(1, 4)
profile : str = 'FUDS'

file_name : str = 'Chemali2017'#'testHyperParams'
model_name: str = 'ModelsUp-1'

# %% [histories]
profile : str = 'DST'
metric : str = 'mae'
attempt : int = 0
titles = {}
data = {}
for profile in profiles:
    names     : list = []
    histories : list = []
    for nLayers in layers:
        nNames = []
        nHistories = []
        for nNeurons in neurons:
            # nNames.append(f'{nLayers}x({nNeurons})-{attempt}')
            nNames.append(f'{nLayers}x({nNeurons})')
            longest = 0
            for a in attempts:
                #! Pick the best
                #! Plot SoC curve at axes[4]
                rows = pd.read_csv(f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                            f'{a}-{profile}/history.csv').shape[0]
                if rows > longest:
                    longest = rows
                    attempt = a
            nHistories.append(
                pd.read_csv(f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                            f'{attempt}-{profile}/history.csv')
                )
        names.append(nNames)
        histories.append(nHistories)
    titles[profile] = names.copy()
    data[profile] = histories.copy()
# %% 
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

def fit_line(profile, file_name, model_name, nLayers, nNeurons):
    thetas = np.zeros(shape=(3,2))
    for a in attempts:
        data = pd.read_csv(f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                        f'{a}-{profile}/history.csv')
        Y = data['train_rms'][2:]
        X = data['Epoch'][2:]
        thetas[a-1,0], thetas[a-1,1] = np.polyfit(X,Y,1)
        # plt.plot(thetas[a-1,0]*X + thetas[a-1,1])
        # plt.plot(Y)
    # print(thetas)
    #y_line = theta[0] * X + theta[1]
    return thetas[np.argmin(thetas[:,0]),:]

file_name : str = 'Chemali2017'#'testHyperParams'
model_name: str = 'ModelsUp-1'
titles = {}
data = {}
profile = 'FUDS'
nLayers = 1
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
MAE = tf.metrics.MeanAbsoluteError()
BestMAE = tf.metrics.MeanAbsoluteError()
for nLayers in range(1,4):
    for nNeurons in [ 131, 262, 524, 1048, 1572]:
        minEpochs = 1000
        dict_MAE = {}
        dict_vMAE = {}
        for profile in ['DST','US06','FUDS']:
            epochs, logits = avr_attempts(profile, file_name, model_name,
                                    nLayers, nNeurons, 'mae')
            # BestEpoch, BestLogits = avr_attempts(profile, file_name, model_name,
            #                         nLayers, nNeurons, 'train_mae')
            #* Minimal Epochs
            if(epochs < minEpochs):
                minEpochs = epochs
            
            #* Getting MAE
            y_true = pd.read_csv(
                            f'Data/validation/{profile}_yt_valid.csv'
                        ).iloc[:,-1]
            MAE.update_state(
                    y_true = y_true,
                    y_pred = logits[:,1:].mean(axis=1)
                )
            dict_MAE[f'{profile}_tr_mae'] = np.array(MAE.result()*100)
            MAE.reset_state()
            
            #! Sample per seconds
            
        #! Recording the results
        a=1
        file = (f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                f'{a}-{profile}/1')
        hist = (f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                f'{a}-{profile}/history.csv')
        #* Get size
        mSize = os.stat(file).st_size/1000000 # bytes

        #* Get time
        tr_rime = np.mean(pd.read_csv(hist)['time(s)'].values[1:])
        while (np.isnan(tr_rime) and a < 4):
            a +=1
            hist = (f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                f'{a}-{profile}/history.csv')
            tr_rime = np.mean(pd.read_csv(hist)['time(s)'].values[1:])

        time_per_samples = np.mean(pd.read_csv(hist)['val_t_s'].values[1:])/pd.read_csv(f'Data/validation/{profile}_y_valid.csv').shape[0]

        #* Determine trend
        try:
            alpha, c = fit_line(profile, file_name, model_name,
                                        nLayers, nNeurons)
        except:
            alpha = 0
            c = 0
        #* Succes rate
        rate = logits[:,1:].shape[1]/3
        TableRecords.loc[len(TableRecords)] = pd.Series(data={
                'Layers' : nLayers, 'Neurons' : nNeurons, 'minEpochs' : minEpochs,
                'alpha': alpha, 'c' : c,    # Line Fit
                'size' : mSize, 'tr_time' : tr_rime, 'Success_rate' : rate,
                'DST_tr_mae(%)' : dict_MAE.get('DST_tr_mae'),
                'US06_tr_mae(%)': dict_MAE.get('US06_tr_mae'),
                'FUDS_tr_mae(%)': dict_MAE.get('FUDS_tr_mae'),
                #'avr_tr_mae(%)' : np.mean(dict_MAE.get('DST_tr_mae'),dict_MAE.get('US06_tr_mae'),dict_MAE.get('FUDS_tr_mae')),
                'Ts_time/sample': time_per_samples
            })
TableRecords['avr_tr_mae(%)'] = np.mean(TableRecords.iloc[:,-3:], axis=1)
# TableRecords.iloc[np.argmin(TableRecords['avr_tr_mae(%)'],3)]
#! Printing bests
criterias = ['avr_tr_mae(%)', 'size', 'alpha', 'Ts_time/sample']
for criteria in criterias:
    print(f'Best by {criteria}')
    TableRecords.iloc[TableRecords.sort_values(by=criteria, ascending=True).index[:3]].head()
# %%
# %% 
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
    return logits

for profile in profiles:
    names : list = []
    train : list = []
    for nLayers in layers:
        nNames = []
        nHistories = []
        for nNeurons in neurons:
            nNames.append(f'{nLayers}x({nNeurons})')
            
            logits = avr_attempts(profile, file_name, model_name,
                                  nLayers, nNeurons)
            nHistories.append(
                logits[:,1:].mean(axis=1)
                )
        names.append(nNames)
        train.append(nHistories)
    titles[profile] = names.copy()
    data[profile] = train.copy()
# %%
profile = 'FUDS'
y_data = pd.read_csv(f'Data/validation/{profile}_yt_valid.csv')

MAE = tf.metrics.MeanAbsoluteError()
lowest, index, lay = 1,0,0
fig, axs = plt.subplots(5,3, figsize=(24,48), dpi=600)
for l in range(3):
    # for i, ax in enumerate(axs):
    for i in range(5):
        MAE.update_state(y_true=y_data.iloc[:,-1], y_pred = data[profile][l][i])
        axs[i, l].plot(y_data.iloc[:,-1],'-',
                    label="Actual", color='#0000ff')
        axs[i, l].plot(data[profile][l][i], '--', label=titles[profile][l][i],
                       color='#ff0000')
        textstr = '\n'.join((
            '$MAE  = {0:.2f}%$'.format(MAE.result()*100, ),
            ))
        axs[i, l].text(0.54, 0.93, textstr, transform=axs[i, l].transAxes, fontsize=30,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        if MAE.result() < lowest:
            index = i
            lay = l
            lowest = MAE.result()
        MAE.reset_state()
        axs[i, l].legend(prop={'size': 32})

print(f'The best model is {titles[profile][lay][index]} with error {lowest*100}')
# %%
# Line Fitter
data = pd.read_csv(f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                        f'{a}-{profile}/history.csv')
Y = data['mae'][3:]
X = data['Epoch'][3:]
theta = np.polyfit(X,Y,1)
print(theta)
y_line = theta[1] + theta[0] * X
plt.plot(X,Y)
plt.plot(X, y_line)
# 0: -0.016
# 1: -0.0107
# 2: -0.005791
# 3: -0.004529

mSize = os.stat(f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                        f'{a}-{profile}/10').st_size/1000000 # bytes
print(f"{mSize:.4}-Mbytes")

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
    fig, axes = plt.subplots(2,2, figsize=(14,12), dpi=600)
    # axes[1][1].set_visible(False)
    # axes[2][0].set_position([0.24,0.125,0.228,0.343])
    minimals : list = []
    indexes : list = []
    for l, ax in enumerate(fig.axes[:-1]):
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
    labels = list(chain.from_iterable(names))
    labels = []
    for l in names:
        for n in l:
            labels.append(n)
        labels.append('')
    
    # _, ax2 = plt.subplots(figsize=(14,12), dpi=600)
    # # hbars = axes[2][0].barh(y_pos, minimals, xerr=indexes, align='center')
    # # [v*100 if v > 0 else None for v in [0, 1 ,2 ,3]]
    ax2 = axes[1][1]
    values = [v*100 for v in minimals]
    y_pos = np.arange(len(labels))
    hbars = ax2.barh(y_pos, values, align='center')
    ax2.set_yticks(y_pos, labels=labels)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('Error(%)')
    ax2.set_title('Minimal Epoch with corresponding index')

    # Label with specially formatted floats
    ax2.bar_label(hbars, fmt='%.2f')
    for i in range(len(y_pos)):
        if(minimals[i] > 0):
            ax2.text(s=indexes[i], x=0.1, y=y_pos[i], verticalalignment="center", color='w')
    # ax2.set_xlim(right=15)  # adjust xlim to fit labels
    idx, value = non_zero_min_idx(values)
    text = f'The minimal set is {labels[idx]} with error: {value}%'
    # print(text)
    fig.suptitle(text)
    fig.tight_layout()
    # #! TODO: Make gaps in the between layers
    # print('TODO: Make gaps between layer')

profile = 'US06'
# plot_bar(neurons, profile, names, histories, 'val_mae', [0, 10])
# plot_bar(neurons, profile, names, histories, 'tes_mae', [0, 10])
# plot_bar(neurons, profile, titles[profile], data[profile], 'mae', [0, 5])
plot_bar(neurons, profile, titles[profile], data[profile], 'tes_mae', [0, 8])
#!!!!!!!!!!!!!!!!!! How do I calculate the trend?
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
# plt.savefig('plt', bbox_inches = "tight")
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
nNeurons : str = '524'
for attempt in range(1, 6):
    hist_path = f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/' \
                f'{attempt}-{profile}/history.csv'
    
    histories.append(
            pd.read_csv(hist_path)
            )
    
    iEpoch, prev_error  = Locate_Best_Epoch(hist_path, 'train_mae')
    
    logits.append(
            pd.read_csv(f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/' \
                f'{attempt}-{profile}/{iEpoch}-train-logits.csv')
        )
    iEpoch, prev_error  = Locate_Best_Epoch(hist_path, 'val_mae')
    logits_val.append(
            pd.read_csv(f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/' \
                f'{attempt}-{profile}/{iEpoch}-valid-logits.csv')
        )
    print(iEpoch)

# %%
#plt.rcParams['figure.figsize'] = [25, 7]
plt.rcParams['figure.figsize'] = [10, 7]
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
# Read throug cycles
# df = pd.read_csv(f'Mods/Model-№1/1xChemali2017-(262)/3-FUDS/history.csv')
# df_c = pd.read_csv(f'Mods/Model-№1/1xChemali2017-(262)/3-FUDS/history-cycles.csv')

df = pd.read_csv(f'Mods/Model-№1/2xChemali2017-(262)/1-FUDS/history.csv')
df_c = pd.read_csv(f'Mods/Model-№1/2xChemali2017-(262)/1-FUDS/history-cycles.csv')

#! IT"S FIXXED!! Use it for future references
metric = 'train_mae'
epochs = len(df[metric])
cycles = 5*epochs
x_cycles = np.linspace(0, epochs, cycles)
x_epoch = np.linspace(x_cycles[4], epochs, epochs)


plt.plot(x_epoch, df[metric])
plt.plot(x_epoch, df_c[metric][4:cycles:5])
plt.plot(x_cycles, df_c[metric][:cycles:])

plt.plot(x_cycles[4::], df_c[metric][4:cycles:])
# %%
length = pd.read_csv(
                f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                f'1-{profile}/1-train-logits.csv').shape[0]
logits = np.empty(shape=(length,1))
for a in range(1,11):
    count = 0
    dir_path = (f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                        f'{a}-{profile}/traiPlots')
    for path in os.scandir(dir_path):
        if path.is_file():
            count += 1

    logits = np.append(logits, pd.read_csv(
                f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/'
                f'{a}-{profile}/{count}-train-logits.csv').iloc[:, -1:].values,
                axis=1)
#* Getting MAE
y_true = pd.read_csv(
                f'/mnt/LibrarySM/SHARED/Data/validation/{profile}_yt_valid.csv'
            ).iloc[:,-1]
MAE = tf.metrics.MeanAbsoluteError()
MAE.update_state(
        y_true = y_true,
        y_pred = logits[:,1:].mean(axis=1)
    )
plt.plot(y_true)
plt.plot(logits[:,1:].mean(axis=1))
plt.title(f'MAE: {MAE.result()*100:.4}% across {logits[:,1:].shape[1]} attempts')