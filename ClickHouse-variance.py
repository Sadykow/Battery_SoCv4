# %%
from clickhouse_driver import Client
import pandas as pd
import numpy as np
import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt

from tensorflow.keras.metrics import mean_absolute_error
# %%
database = 'mln_base'
client = Client(host='10.137.0.39', port=9000, 
                database=database, user='tf', password='TF28')
mpl.rcParams['font.family'] = 'Bender'
client.execute('SHOW DATABASES')

def show_tables():
    return client.execute(f'SHOW TABLES FROM {database};')
def describe_table(table : str):
    return client.execute(f'DESCRIBE TABLE {database}.{table} SETTINGS describe_include_subcolumns=1;')
def describe_simple_table(table : str):
    return client.execute(f'DESCRIBE TABLE {database}.{table};')
def get_table(table : str):
    return client.execute(f'SELECT * FROM {database}.{table}')

#? 1) Mapper table ['File', 'ModelID', 'Profile', 'Attempt', 'Name' 'Hist_plots(train, valid)',
#?                                                    'Best_plots(train, valid, test1, test2)']
#? 2) Histories table ['File', 'Epoch', 'mae', 'rmse', 'rsquare', 'time_s', 'learn_r',
#?                                                                       'train()', 'val()', 'tes()']
#? 3) Faulty Histories ['File', 'Epoch', 'attempt', 'mae', 'time_s', learn_rate, train ()]
#? 4) Logits table logits_(train, valid,test) ['File', 'Epoch', logits()]
#? 5) Models table ['File', 'Epochs', 'Latest_Model', 'TR_Model', 'VL_Model', 'TS_Model']
#? 6) Full train logits table ['File', 'Epochs', 'Logits']
mapper : str = 'mapper'
histores : str = 'histories'
faulties : str = 'faulties'
models : str = 'models'
accuracies : str = 'logits_full_train'
ff_accuracies : str = 'logits_ff_train'
# %%
#[markdown] This is the table generator. Procudes accuraced relatively to what model where tested
name : str = 'Sadykov2022'
tested_p : str = 'DST'
trained_p : str = 'US06'

# Get N of attempts
N_attempt = client.execute(
    f"SELECT MAX(Attempt) FROM {database}.{mapper} "
    f"WHERE (Name = '{name}') AND Profile='{trained_p}'")[0][0]

# query=(f"SELECT train.mae FROM {database}.{histores} WHERE ({database}.{histores}.id in ("
#             f"SELECT id FROM {database}.{mapper} WHERE (Name = '{name}') AND Profile='{trained_p}'"
#             f"AND Attempt IN UNNEST(['1', '2', '3', '4']) ORDER BY Attempt));")
# client.execute(query)

# history = client.query_dataframe(
#         query=(f"SELECT train.mae FROM {database}.{histores} WHERE ({database}.{histores}.id in ("
#             f"SELECT id FROM {database}.{mapper} WHERE (Name = '{name}') AND Profile='{trained_p}'"
#             f"AND Attempt IN ('1', '2', '3', '4') ORDER BY Attempt));")
#     )

attempt : int = 1
data = pd.DataFrame()
while (attempt <= N_attempt):
    history = client.query_dataframe(
        query=(f"SELECT train.mae FROM {database}.{histores} WHERE ({database}.{histores}.id in ("
            f"SELECT id FROM {database}.{mapper} WHERE (Name = '{name}') AND Profile='{trained_p}'"
            f"AND Attempt = '{attempt}'));")
            )
    print(f'Histories: {history.shape}')
    plt.plot(history.values)
    # plt.ylim([0, 0.11])
    # data = data.append([v[0] for v in history.values], ignore_index=True)
    # data[f'{attempt}'] = [v[0] for v in history.values]
    data = pd.concat([data,
                      pd.DataFrame([v[0] for v in history.values], columns=[f'{attempt}'])
                    ], axis=1)
    attempt +=1
# np.asarray(data)
# %%
fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
ax1.set_title(f'Model №6 - General variance of MAE over\n training history across 10 attempts',
              fontsize=36)

ax1.plot(data.max(axis=1,skipna=True)*100, '--', label='Max line', color='#ff0000')
ax1.plot(data.min(axis=1,skipna=True)*100, '--', label='Min line', color='#ff0000')
ax1.plot(data.mean(axis=1,skipna=True)*100, '-', label='Average learning', color='#0000ff')
ax1.set_xlabel("Epochs", fontsize=32)
ax1.set_ylabel("Error (%)", fontsize=32)
ax1.legend(prop={'size': 32})
ax1.tick_params(axis='both', labelsize=28)
ax1.set_ylim([-0.1,11])
ax1.set_xlim([-1,20])
fig.tight_layout()
fig.savefig(f'Modds/M6-variance.svg')
# %%

database = 'ml_base'
name : str = 'Chemali2017'
trained_p : str = 'DST'

# Get N of attempts
N_attempt = client.execute(
    f"SELECT MAX(Attempt) FROM {database}.{mapper} "
    f"WHERE (Name = '{name}') AND Profile='{trained_p}'")[0][0]

attempt : int = 1
data = pd.DataFrame()
while (attempt <= N_attempt):
    history = client.query_dataframe(
        query=(f"SELECT train.mae FROM {database}.{histores} WHERE ({database}.{histores}.id in ("
            f"SELECT id FROM {database}.{mapper} WHERE (Name = '{name}') AND Profile='{trained_p}'"
            f"AND Attempt = '{attempt}'));")
            )
    print(f'Histories: {history.shape}')
    plt.plot(history.values)
    plt.ylim([0, 0.11])
    # data = data.append([v[0] for v in history.values], ignore_index=True)
    # data[f'{attempt}'] = [v[0] for v in history.values]
    data = pd.concat([data,
                      pd.DataFrame([v[0] for v in history.values], columns=[f'{attempt}'])
                    ], axis=1)
    attempt +=1

fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
ax1.set_title(f'Model №1 - General variance of MAE over\n training history across 10 attempts',
              fontsize=36)

ax1.plot(data.max(axis=1,skipna=True)*100, '--', label='Max line', color='#ff0000')
ax1.plot(data.min(axis=1,skipna=True)*100, '--', label='Min line', color='#ff0000')
ax1.plot(data.mean(axis=1,skipna=True)*100, '-', label='Average learning', color='#0000ff')
ax1.set_xlabel("Epochs", fontsize=32)
ax1.set_ylabel("Error (%)", fontsize=32)
ax1.legend(prop={'size': 32})
ax1.tick_params(axis='both', labelsize=28)
ax1.set_ylim([-0.1,11])
ax1.set_xlim([-1,20])
fig.tight_layout()
fig.savefig(f'Modds/M1-variance.svg')
# %%

database = 'ml_base'
name : str = 'TadeleMamo2020'
trained_p : str = 'FUDS'

# Get N of attempts
N_attempt = client.execute(
    f"SELECT MAX(Attempt) FROM {database}.{mapper} "
    f"WHERE (Name = '{name}') AND Profile='{trained_p}'")[0][0]

attempt : int = 1
data = pd.DataFrame()
while (attempt <= N_attempt):
    history = client.query_dataframe(
        query=(f"SELECT train.mae FROM {database}.{histores} WHERE ({database}.{histores}.id in ("
            f"SELECT id FROM {database}.{mapper} WHERE (Name = '{name}') AND Profile='{trained_p}'"
            f"AND Attempt = '{attempt}'));")
            )
    print(f'Histories: {history.shape}')
    plt.plot(history.values)
    plt.ylim([0, 0.11])
    # data = data.append([v[0] for v in history.values], ignore_index=True)
    # data[f'{attempt}'] = [v[0] for v in history.values]
    data = pd.concat([data,
                      pd.DataFrame([v[0] for v in history.values], columns=[f'{attempt}'])
                    ], axis=1)
    attempt +=1

fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
ax1.set_title(f'Model №3 - General variance of MAE over\n training history across 10 attempts',
              fontsize=36)

ax1.plot(data.max(axis=1,skipna=True)*100, '--', label='Max line', color='#ff0000')
ax1.plot(data.min(axis=1,skipna=True)*100, '--', label='Min line', color='#ff0000')
ax1.plot(data.mean(axis=1,skipna=True)*100, '-', label='Average learning', color='#0000ff')
ax1.set_xlabel("Epochs", fontsize=32)
ax1.set_ylabel("Error (%)", fontsize=32)
ax1.legend(prop={'size': 32})
ax1.tick_params(axis='both', labelsize=28)
ax1.set_ylim([-0.1,11])
ax1.set_xlim([-1,20])
fig.tight_layout()
fig.savefig(f'Modds/M3-variance.svg')
# %%
