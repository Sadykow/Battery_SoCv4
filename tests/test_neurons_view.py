# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # File read

model  : str = 'LSTM'
layers : str = ''
attempt: int = 1
neurons: list = [8, 16, 32, 65, 131, 262, 524]

location : str = f'Models/{model}{layers}-Attempt{attempt}'

#! Make a list of DFs and go over them
DFs = []
for neu in neurons:
    hist_file: str = f'{model}{layers}-history-{neu}.csv'
    DFs.append(pd.read_csv(f'{location}/{hist_file}'))

x : np.ndarray = np.arange(10)
titles : list = DFs[0].columns
# %%
# Mean Absolute Error
fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(12,5))
title : str = 'mean_absolute_error'
neu : int = 0
for df in DFs[::]:
    ax1.plot(x, df[title]*100, label=f'{model}-{neurons[neu]}')
    neu += 1
ax1.legend()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Error (%)')
ax1.set_title(title)
title : str = 'val_mae'
neu : int = 0
for df in DFs[::]:
    ax2.plot(x, df[title]*100, label=f'{model}-{neurons[neu]}')
    neu += 1
ax2.legend()
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Error (%)')
ax2.set_title(title)

# %%
# RMSE
fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(12,5))
title : str = 'root_mean_squared_error'
neu : int = 0
for df in DFs[::]:
    ax1.plot(x, df[title]*100, label=f'{model}-{neurons[neu]}')
    neu += 1
ax1.legend()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Error (%)')
ax1.set_title(title)
title : str = 'val_rmse'
neu : int = 0
for df in DFs[::]:
    ax2.plot(x, df[title]*100, label=f'{model}-{neurons[neu]}')
    neu += 1
ax2.legend()
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Error (%)')
ax2.set_title(title)

# %%
# R2
fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(12,5))
title : str = 'r_square'
neu : int = 0
for df in DFs[::]:
    ax1.plot(x, df[title]*100, label=f'{model}-{neurons[neu]}')
    neu += 1
ax1.legend()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Error (%)')
ax1.set_title(title)
title : str = 'val_r2'
neu : int = 0
for df in DFs[::]:
    ax2.plot(x, df[title]*100, label=f'{model}-{neurons[neu]}')
    neu += 1
ax2.legend()
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Error (%)')
ax2.set_title(title)

# %%
means = []
params = []
neu : int = 0
for df in DFs[::]:
    means.append(np.mean(df['Time(s)']))
    params.append(np.mean(df['N_Params']))
    print(
        f'Neurons: {neurons[neu]}\t{means[neu]:.2f} \t '
        f'Params: {params[neu]}  \t '
        f'Size: '
        )
    neu += 1
plt.figure()
plt.plot(neurons, means, '-x')
plt.xticks(neurons)
plt.xlabel('neurons')
plt.ylabel('Time mean')

plt.figure()
plt.plot(neurons, params, '-x')
plt.xticks(neurons)
plt.xlabel('neurons')
plt.ylabel('Params')

# %%
# Average of 3
model  : str = 'LSTM'
layers : str = ''
attempt: int = 2
neurons: list = [8, 16, 32, 65, 131, 262, 524]

#! Make a list of DFs and go over them
ATs = []
for attempt in range(1,4):
    location : str = f'Models/{model}{layers}-Attempt{attempt}'
    DFs = []
    for neu in neurons:
        hist_file: str = f'{model}-history-{neu}.csv'
        DFs.append(pd.read_csv(f'{location}/{hist_file}'))
    ATs.append(DFs)
x : np.ndarray = np.arange(10)
titles : list = DFs[0].columns

# %%
# Mean Absolute Error
fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(12,5))
title : str = 'mean_absolute_error'
for neu in range(len(neurons)):
    average = np.expand_dims(ATs[0][neu][title], axis=1)
    for i in range(1,3):
        average = np.append(
                    average,
                    np.expand_dims(ATs[i][neu][title], axis=1),
                    axis=1
                )
    average = np.mean(average, axis=1)
    ax1.plot(x, average*100, label=f'{model}-{neurons[neu]}')

ax1.legend()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Error (%)')
ax1.set_title(title)
title : str = 'val_mae'
for neu in range(len(neurons)):
    average = np.expand_dims(ATs[0][neu][title], axis=1)
    for i in range(1,3):
        average = np.append(
                    average,
                    np.expand_dims(ATs[i][neu][title], axis=1),
                    axis=1
                )
    # print(average)
    average = np.mean(average, axis=1)
    ax2.plot(x, average*100, label=f'{model}-{neurons[neu]}')
ax2.legend()
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Error (%)')
ax2.set_title(title)

# %%
# Root Mean Squared Error
fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(12,5))
title : str = 'root_mean_squared_error'
for neu in range(len(neurons)):
    average = np.expand_dims(ATs[0][neu][title], axis=1)
    for i in range(1,3):
        average = np.append(
                    average,
                    np.expand_dims(ATs[i][neu][title], axis=1),
                    axis=1
                )
    average = np.mean(average, axis=1)
    ax1.plot(x, average*100, label=f'{model}-{neurons[neu]}')

ax1.legend()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Error (%)')
ax1.set_title(title)
title : str = 'val_rmse'
for neu in range(len(neurons)):
    average = np.expand_dims(ATs[0][neu][title], axis=1)
    for i in range(1,3):
        average = np.append(
                    average,
                    np.expand_dims(ATs[i][neu][title], axis=1),
                    axis=1
                )
    average = np.mean(average, axis=1)
    ax2.plot(x, average*100, label=f'{model}-{neurons[neu]}')
ax2.legend()
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Error (%)')
ax2.set_title(title)

# %%
means = []
params = []
# neu : int = 0
# for df in DFs[::]:
#     means.append(np.mean(df['Time(s)']))
#     params.append(np.mean(df['N_Params']))
#     print(
#         f'Neurons: {neurons[neu]}\t{means[neu]:.2f} \t '
#         f'Params: {params[neu]}  \t '
#         f'Size: '
#         )
#     neu += 1
title : str = 'Time(s)'
for neu in range(len(neurons)):
    av_time = np.expand_dims(ATs[0][neu][title], axis=1)
    av_time = np.expand_dims(ATs[0][neu][title], axis=1)
    for i in range(1,3):
        av_time = np.append(
                    av_time,
                    np.expand_dims(ATs[i][neu][title], axis=1),
                    axis=1
                )
    # print(average)
    av_time = np.mean(av_time)
    print(
        f'Average Time: \t{av_time:.2f} \t '
        # f'Params: {params[neu]}  \t '
        # f'Size: '
        )

# plt.figure()
# plt.plot(neurons, means, '-x')
# plt.xticks(neurons)
# plt.xlabel('neurons')
# plt.ylabel('Time mean')

# plt.figure()
# plt.plot(neurons, params, '-x')
# plt.xticks(neurons)
# plt.xlabel('neurons')
# plt.ylabel('Params')

# %%
# alpha   = [10,  5, 2.5, 1.25, 0.625, 0.3225, 0.15625]
# neurons = [ 8, 16,  32,   65,   131,    262,     524]

# ???Layers???


#                                  LSTM / (GPU)
# alhpa | Neurons |      Atempt 1   |      Atempt 2    |  Time | Size    |
#       |         | MAE | RMSE | R2 |  MAE | RMSE | R2 |       |         |
# 10    | 8       | 0.09|           | 0.1              | 50s   | 3.44KB  |



# %%
