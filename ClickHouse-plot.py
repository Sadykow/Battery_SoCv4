# %%
from clickhouse_driver import Client
import pandas as pd
import numpy as np
import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt

from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
import tensorflow as tf
from tensorflow.keras.metrics import mean_absolute_error, MeanAbsoluteError, RootMeanSquaredError
from tensorflow_addons.metrics import RSquare

physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(
#                         device=physical_devices[0], enable=True)
# tf.config.experimental.set_memory_growth(
#                         device=physical_devices[1], enable=True)
# %%
database = 'ml_base'
client = Client(host='192.168.1.254', port=9000, 
                database=database, user='tf', password='TF28')

client.execute('SHOW DATABASES')
# %%
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

# %% [full train logits]
# describe_simple_table(accuracies)
with open(f'Data/np-data/DST/train.npy', 'rb') as f:
    _ = np.load(f)
    DST_y = np.load(f)[:,0,0]
with open(f'Data/np-data/US06/train.npy', 'rb') as f:
    _ = np.load(f)
    US_y = np.load(f)[:,0,0]
with open(f'Data/np-data/FUDS/train.npy', 'rb') as f:
    _ = np.load(f)
    y_train = np.load(f)[:,0,0]
    split = int(y_train.shape[0]/3)
    Y = y_train[0:split]
    Y = np.append(Y, y_train[split:2*split])
    Y = np.append(Y, y_train[2*split:3*split])
    FUDS_y = Y.copy()

Ys = { 'DST' : DST_y, 'US06' : US_y, 'FUDS' : FUDS_y}
# %%
#[markdown] This is the table generator. Procudes accuraced relatively to what model where tested
name : str = 'BinXiao2021'
tested_p : str = 'DST'
trained_p : str = 'FUDS'

MAE = MeanAbsoluteError()
RMSE = RootMeanSquaredError()
RS = RSquare(y_shape=(1,), dtype=tf.float32)
# with open(f'Data/np-data/{tested_p}/train.npy', 'rb') as f:
#     _ = np.load(f)
#     y_train = np.load(f)[:,0,0]
final_print = ''
for trained_p in ['DST', 'FUDS']:
    line = ''
    for tested_p in ['DST', 'US06', 'FUDS']:
        logits = client.query_dataframe(
            query=f"SELECT Logits FROM {database}.{accuracies} WHERE ({database}.{accuracies}.id in ("
                f"SELECT id FROM {database}.{mapper} WHERE (Name = '{name}') AND Profile='{trained_p}'"
                f") AND Profile = '{tested_p}');"
                )

        print(f'Logits: {logits.shape}')
        errors = []
        for i in range(logits.shape[0]):
            errors.append(mean_absolute_error(
                y_true=Ys[tested_p], y_pred=np.array(logits.loc[i].values[0])
            ).numpy())

        df_errors = pd.DataFrame(data=errors, columns=['mae'])
        df_errors = df_errors.reset_index()
        df_errors = df_errors.sort_values('mae')
        print(df_errors)
        indexes = df_errors.sort_values('mae')['index'][:].values
        # indexes = df_errors['index'][
        #     df_errors['mae'] < np.mean(df_errors['mae'])
        #     ].values

        # plt.plot(logits.loc[0].values[0][:15000])
        # average = np.array(logits.loc[0].values[0])
        average = np.zeros(shape=len(logits.loc[0].values[0]))
        for i in indexes:
            average += logits.loc[i].values[0]
        average = average / len(indexes)

        MAE.update_state(y_true=Ys[tested_p],
                            y_pred=average)
        RMSE.update_state(y_true=Ys[tested_p],
                            y_pred=average)
        RS.update_state(y_true=np.expand_dims(Ys[tested_p],axis=1),
                        y_pred=np.expand_dims(average,axis=1))
        mae = MAE.result().numpy()*100
        rmse = RMSE.result().numpy()*100
        rs = RS.result().numpy()*100

        plt.figure()
        plt.plot(Ys[tested_p][:15000])
        plt.plot(average[:15000])
        plt.title(mean_absolute_error(
                    y_true=Ys[tested_p],
                    y_pred=average
                ).numpy()
            )
        plt.show()
        # print(f'Results: {mae:.2f} & {rmse:.2f} & {rs:.2f} ')
        line += f'& {mae:.2f} & {rmse:.2f} & {rs:.2f} '
        MAE.reset_states()
        RMSE.reset_states()
        RS.reset_states()
    print(trained_p, end=' ')
    print(line + '\\\ ')
    final_print += f'{trained_p} {line} \\\ \n'
print('- -'*25)
print(final_print)
# %%
name : str = 'BinXiao2021'
trained_p : str = 'DST'
tested_p : str = 'DST'

MAE = MeanAbsoluteError()
RMSE = RootMeanSquaredError()
RS = RSquare(y_shape=(1,), dtype=tf.float32)

# %%
line = ''
l_type = 'test'
f_type = 'test'

with open(f'Data/np-data/DST/{f_type}.npy', 'rb') as f:
    _ = np.load(f)
    DST_y = np.load(f)[:,0,0]
with open(f'Data/np-data/US06/{f_type}.npy', 'rb') as f:
    _ = np.load(f)
    US_y = np.load(f)[:,0,0]
with open(f'Data/np-data/FUDS/{f_type}.npy', 'rb') as f:
    _ = np.load(f)
    FUDS_y = np.load(f)[:,0,0]

Ys = { 'DST' : DST_y, 'US06' : US_y, 'FUDS' : FUDS_y}

for trained_p in ['DST', 'US06', 'FUDS']:
    errors = []
    arr_logits = np.zeros(shape=(Ys[trained_p].shape[0], 10))
    for a in range(1, 11):
        epoch = client.execute(f"SELECT MAX(Epoch) FROM {database}.{histores} WHERE ({database}.{histores}.id in ("
                    f"SELECT id FROM {database}.{mapper} WHERE (Name = '{name}') AND Profile='{trained_p}' AND Attempt={a}"
                    ") );")[0][0]

        #! For tests use arr_logits[:-400,a-1] - hell knows why
        arr_logits[:-400,a-1] = client.execute(
            query=f"SELECT Logits FROM {database}.logits_{l_type} WHERE ({database}.logits_{l_type}.id in ("
                f"SELECT id FROM {database}.{mapper} WHERE (Name = '{name}') AND Profile='{trained_p}' AND Attempt={a}"
                f") AND Epoch = {epoch});" # Max(Epoch) for an attempt
                )[0][0]
        errors.append(mean_absolute_error(
            y_true=Ys[trained_p], y_pred=arr_logits[:,a-1]
        ).numpy())
    df_errors = pd.DataFrame(data=errors, columns=['mae'])
    print(df_errors.sort_values('mae'))
    indexes = df_errors[
        df_errors['mae'] < np.mean(df_errors['mae'])
        ].index

    average = arr_logits[:,indexes].mean(axis=1)
    MAE.update_state(y_true=Ys[trained_p],
                        y_pred=average)
    RMSE.update_state(y_true=Ys[trained_p],
                        y_pred=average)
    RS.update_state(y_true=np.expand_dims(Ys[trained_p],axis=1),
                    y_pred=np.expand_dims(average,axis=1))
    mae = MAE.result().numpy()*100
    rmse = RMSE.result().numpy()*100
    rs = RS.result().numpy()*100

    plt.figure()
    plt.plot(Ys[trained_p])    
    plt.plot(average)
    plt.title(mean_absolute_error(
                y_true=Ys[trained_p],
                y_pred=average
            ).numpy()
        )

    print(f'Results: & {mae:.2f} & {rmse:.2f} & {rs:.2f} ')
    line += f'& {mae:.2f} & {rmse:.2f} & {rs:.2f} '
    MAE.reset_states()
    RMSE.reset_states()
    RS.reset_states()
print(line + '\\\ ')
# %% [average histories]
#describe_simple_table(histores)
#! Making plots
def history_plot(profile : str, file_name : str, model_loc : str,
                 df : pd.DataFrame, save_plot : bool = False,
                 metrics : list = ['mae', 'train_mae',
                                   'rmse', 'train_rms'],
                 plot_file_name : str = 'history.svg') -> None:
  fig, ax1 = plt.subplots(1, figsize=(14,12), dpi=600)
#   fig, ax1 = plt.subplots(1)
  fig.suptitle(f'{file_name} - {profile} training benchmark',
              fontsize=36)
  
  # Plot MAE subfigure
  ax1.plot(df[metrics[0]]*100, '-o',
      label="Training", color='#0000ff')
  ax1.plot(df[metrics[1]]*100, '--o',
      label="Testing", color='#ff0000')
  ax1.set_xlabel("Epochs", fontsize=32)
  ax1.set_ylabel("Error (%)", fontsize=32)

  # Plot RMSE subfigure
  ax1.set_ylabel("Error (%)", fontsize=32)
  ax1.legend(prop={'size': 32})
  ax1.tick_params(axis='both', labelsize=28)

  # Tighting the layot
  ax1.set_title(f"10-attempts average Mean Absoulute Error", fontsize=36)
  ax1.set_ylim([-0.1,6.1])
#   ax1.set_ylim([79,101])
#   ax2.set_ylim([-0.1,11])
  fig.tight_layout()
  
  # Saving figure and cleaning Memory from plots
  if save_plot:
    fig.savefig(f'{model_loc}{plot_file_name}')
#   fig.clf()

def predicting_plot(profile : str, file_name : str, model_loc : str,
                    model_type : str,  iEpoch : str,
                    Y : np.ndarray, PRED : np.ndarray, RMS : np.ndarray,
                    val_perf : np.ndarray, TAIL : int,
                    save_plot : bool = False, RMS_plot : bool = True) -> None:
  def format_SoC(value, _):
    return int(value*100)
  # Time range
  test_time = np.linspace(0, PRED.shape[0]/60, PRED.shape[0])
  
  # instantiate the first axes
  fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
  fig.suptitle(f"{file_name} {model_type}. {profile}-trained",
              fontsize=36)
  ax1.plot(test_time[:TAIL:], Y[::,], '-',
          label="Actual", color='#0000ff')
  ax1.plot(test_time[:TAIL:],
          PRED, '--',
          label="Prediction", color='#ff0000')
  # ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
  ax1.set_xlabel("Time Slice (min)", fontsize=32)
  ax1.set_ylabel("SoC (%)", fontsize=32)
  
  # instantiate a second axes that shares the same x-axis
  if RMS_plot:
    ax2 = ax1.twinx()
    ax2.plot(test_time[:TAIL:],
          RMS,
          label="ABS error", color='#698856')
    ax2.fill_between(test_time[:TAIL:],
          RMS,
          color='#698856')
    ax2.set_ylabel('Error', fontsize=32, color='#698856')
    ax2.tick_params(axis='y', labelcolor='#698856', labelsize=28)
    ax2.set_ylim([-0.1,1.6])
    ax2.legend(loc='center right', bbox_to_anchor=(1.0,0.80), prop={'size': 32})
  ax1.set_title(
      f"10-attempts average SoC(%) prediction",
      fontsize=36)
  ax1.legend(prop={'size': 32})
  ax1.tick_params(axis='both', labelsize=28)
  ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_SoC))
  ax1.set_ylim([-0.1,1.2])
  fig.tight_layout()

  # Put the text box with performance results.
  # textstr = '\n'.join((
  #     r'$MAE =%.2f$'  % (val_perf[1]*100, ),
  #     r'$RMSE=%.2f$'  % (val_perf[2]*100, ),
  #     r'$R^2 =%.2f$'  % (val_perf[3]*100, )))
  textstr = '\n'.join((
       '$MAE  = {0:.2f}%$'.format(val_perf[1], ),
       '$RMSE = {0:.2f}%$'.format(val_perf[2], ),
       '$R2  = {0:.2f}%$'.format(val_perf[3], ) ))
  ax1.text(0.66, 0.74, textstr, transform=ax1.transAxes, fontsize=30,
          verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
  
  # Saving figure and cleaning Memory from plots
  if save_plot:
    fig.savefig(f'{model_loc}{profile}-{iEpoch}.svg')
#   fig.clf()
# %%
ModelID = 2

trained_p : str = 'DST'

criteria = 'mae'
attempts = range(3,4)

a = 2
df  = client.query_dataframe(
        query=f"SELECT {criteria} FROM {database}.{histores} WHERE ( {database}.{histores}.id in ("
            f"SELECT id FROM {database}.{mapper} WHERE (ModelID = '{ModelID}') AND Profile='{trained_p}' AND Attempt = {a}"
            f") );"
        )
for a in attempts:
    df = pd.concat([
        df,
        client.query_dataframe(
            query=f"SELECT {criteria} FROM {database}.{histores} WHERE ( {database}.{histores}.id in ("
                f"SELECT id FROM {database}.{mapper} WHERE (ModelID = '{ModelID}') AND Profile='{trained_p}' AND Attempt = {a}"
                f") );"
            )
        ], axis = 1)

criteria = 'tes.mae'

a = 2
df2  = client.query_dataframe(
        query=f"SELECT {criteria} FROM {database}.{histores} WHERE ( {database}.{histores}.id in ("
            f"SELECT id FROM {database}.{mapper} WHERE (ModelID = '{ModelID}') AND Profile='{trained_p}' AND Attempt = {a}"
            f") );"
        )
for a in attempts:
    df2 = pd.concat([
        df2,
        client.query_dataframe(
            query=f"SELECT {criteria} FROM {database}.{histores} WHERE ( {database}.{histores}.id in ("
                f"SELECT id FROM {database}.{mapper} WHERE (ModelID = '{ModelID}') AND Profile='{trained_p}' AND Attempt = {a}"
                f") );"
            )
        ], axis = 1)

# def history_plot(profile : str, file_name : str, model_loc : str,
#                  df : pd.DataFrame, save_plot : bool = False,
#                  metrics : list = ['mae', 'train_mae',
#                                    'rmse', 'train_rms'],
#                  plot_file_name : str = 'history.svg') -> None:
#   fig, ax1 = plt.subplots(1, figsize=(14,12), dpi=600)
# #   fig, ax1 = plt.subplots(1)
#   fig.suptitle(f'{file_name} - {profile} training benchmark',
#               fontsize=36)
  
#   # Plot MAE subfigure
#   ax1.plot(df[metrics[0]]*100, '-o',
#       label="Training", color='#0000ff')
#   ax1.plot(df[metrics[1]]*100, '--o',
#       label="Testing", color='#ff0000')
#   ax1.set_xlabel("Epochs", fontsize=32)
#   ax1.set_ylabel("Error (%)", fontsize=32)

#   # Plot RMSE subfigure
#   ax1.set_ylabel("Error (%)", fontsize=32)
#   ax1.legend(prop={'size': 32})
#   ax1.tick_params(axis='both', labelsize=28)

#   # Tighting the layot
#   ax1.set_title(f"10-attempts average Mean Absoulute Error", fontsize=36)
#   ax1.set_ylim([-0.1,6.1])
# #   ax1.set_ylim([79,101])
# #   ax2.set_ylim([-0.1,11])
#   fig.tight_layout()
  
#   # Saving figure and cleaning Memory from plots
#   if save_plot:
#     fig.savefig(f'{model_loc}{plot_file_name}')
# #   fig.clf()

#plt.plot(df)
# plt.plot(df.mean(axis=1))

hisotory_df = pd.DataFrame(data={
        'mae' : df.mean(axis=1),
        'test' : df2.mean(axis=1)
    })

# i = 9
# hisotory_df = pd.DataFrame(data={
#         'mae' : df,
#         'test' : df2
#     })
# plt.plot(hisotory_df)
history_plot(trained_p, f'Model №2', 'Modds/tmp/', hisotory_df,
                save_plot=True,
                metrics=['mae','test'],
                #plot_file_name=f'M{ModelID}-history-{trained_p}-mae.svg')
                plot_file_name=f'M{ModelID}-history-{trained_p}-mae.svg')


# fig, ax1 = plt.subplots(1, figsize=(14,12), dpi=600)
# #   fig, ax1 = plt.subplots(1)
# fig.suptitle(f'10 models - FUDS training benchmark',
#             fontsize=36)

# # Plot MAE subfigure
# ax1.plot(df*100, '-o',
#     label="Training", color='#0000ff')
# ax1.plot(df2*100, '--o',
#     label="Testing", color='#ff0000')
# ax1.set_xlabel("Epochs", fontsize=32)
# ax1.set_ylabel("Error (%)", fontsize=32)

# # Plot RMSE subfigure
# ax1.set_ylabel("Error (%)", fontsize=32)
# # ax1.legend(prop={'size': 32})
# ax1.tick_params(axis='both', labelsize=28)

# # Tighting the layot
# ax1.set_ylim([-0.1,6.1])
# fig.tight_layout()

# # Saving figure and cleaning Memory from plots
# fig.savefig('Modds/tmp/DEMO2-history-FUDS-mae.svg')

# %%
###
### Results averaging demonstration for performance verification
### 3 subplots with a single, multiple and average of 10
ModelID = 4

trained_p : str = 'FUDS'

criteria = 'Logits'

a = 1
df  = pd.DataFrame(data = {
    'Y' : client.query_dataframe(
        query=f"SELECT {criteria} FROM {database}.logits_train WHERE ("
            f"{database}.logits_train.id in ("
                f"SELECT id FROM {database}.{mapper} WHERE (ModelID = '{ModelID}') "
                f"AND Profile='{trained_p}' AND Attempt = {a}"
            f") "
            f" );"
        ).iloc[-1, 0] })
for a in range(2,11):
    df = pd.concat([
        df,
        pd.DataFrame(data = {
        'Y' : client.query_dataframe(
            query=f"SELECT {criteria} FROM {database}.logits_train WHERE ( {database}.logits_train.id in ("
                f"SELECT id FROM {database}.{mapper} WHERE (ModelID = '{ModelID}') AND Profile='{trained_p}' AND Attempt = {a}"
                f") );"
            ).iloc[-1, 0] })
        ], axis = 1)

hisotory_df = pd.DataFrame(data={
        'mae' : df.mean(axis=1),
        'true': FUDS_y
    })

history_plot(trained_p, f'10 models', 'Modds/tmp/', hisotory_df,
                save_plot=False,
                metrics=['mae'],
                #plot_file_name=f'M{ModelID}-history-{trained_p}-mae.svg')
                plot_file_name=f'M{ModelID}-logits_train-{trained_p}-mae.svg')


# %%
# def predicting_plot(profile : str, file_name : str, model_loc : str,
#                     model_type : str,  iEpoch : str,
#                     Y : np.ndarray, PRED : np.ndarray, RMS : np.ndarray,
#                     val_perf : np.ndarray, TAIL : int,
#                     save_plot : bool = False, RMS_plot : bool = True) -> None:
#   def format_SoC(value, _):
#     return int(value*100)
#   # Time range
#   test_time = np.linspace(0, PRED.shape[0]/60, PRED.shape[0])
  
#   # instantiate the first axes
#   fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
# #   fig.suptitle(f"{file_name} {model_type}. {profile}-trained",
# #               fontsize=36)
#   ax1.plot(test_time[:TAIL:], Y[::,], '-',
#           label="Actual", color='#0000ff')
#   ax1.plot(test_time[:TAIL:],
#           PRED, '--',
#           label="Prediction", color='#ff0000')
#   # ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
#   ax1.set_xlabel("Time Slice (min)", fontsize=32)
#   ax1.set_ylabel("SoC (%)", fontsize=32)
  
#   # instantiate a second axes that shares the same x-axis
#   if RMS_plot:
#     ax2 = ax1.twinx()
#     for i in range(10):
#         RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
#                 Ys[trained_p]-arr_logits[:,i])))
#         ax2.plot(test_time[:TAIL:],
#             RMS,
#             label="ABS error", color='#698856')
#         ax2.fill_between(test_time[:TAIL:],
#             RMS,
#             color='#698856')
#     ax2.set_ylabel('Error', fontsize=32, color='#698856')
#     ax2.tick_params(axis='y', labelcolor='#698856', labelsize=28)
#     ax2.set_ylim([-0.1,1.6])
#     # ax2.legend(loc='center right', bbox_to_anchor=(1.0,0.80), prop={'size': 32})
#   ax1.set_title(
#       f"10 model - FUDS SoC(%) benchmark",
#       fontsize=36)
# #   ax1.legend(prop={'size': 32})
#   ax1.tick_params(axis='both', labelsize=28)
#   ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_SoC))
#   ax1.set_ylim([-0.1,1.2])
#   fig.tight_layout()

#   # Put the text box with performance results.
#   # textstr = '\n'.join((
#   #     r'$MAE =%.2f$'  % (val_perf[1]*100, ),
#   #     r'$RMSE=%.2f$'  % (val_perf[2]*100, ),
#   #     r'$R^2 =%.2f$'  % (val_perf[3]*100, )))
#   textstr = '\n'.join((
#        '$MAE  = {0:.2f}%$'.format(val_perf[1], ),
#        '$RMSE = {0:.2f}%$'.format(val_perf[2], ),
#        '$R2  = {0:.2f}%$'.format(val_perf[3], ) ))
# #   ax1.text(0.66, 0.74, textstr, transform=ax1.transAxes, fontsize=30,
# #           verticalalignment='top',
# #           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
  
#   # Saving figure and cleaning Memory from plots
#   if save_plot:
#     fig.savefig(f'{model_loc}{profile}-{iEpoch}.svg')
# #   fig.clf()
ModelID = 2
attempts = range(1,4)
# l_type = 'train'
# f_type = 't_valid'

l_type = 'test'
f_type = 'test'
trained_p : str = 'FUDS'

with open(f'Data/np-data/DST/{f_type}.npy', 'rb') as f:
    _ = np.load(f)
    DST_y = np.load(f)[:,0,0]
with open(f'Data/np-data/US06/{f_type}.npy', 'rb') as f:
    _ = np.load(f)
    US_y = np.load(f)[:,0,0]
with open(f'Data/np-data/FUDS/{f_type}.npy', 'rb') as f:
    _ = np.load(f)
    FUDS_y = np.load(f)[:,0,0]

Ys = { 'DST' : DST_y, 'US06' : US_y, 'FUDS' : FUDS_y}

MAE = MeanAbsoluteError()
RMSE = RootMeanSquaredError()
RS = RSquare(y_shape=(1,), dtype=tf.float32)

errors = []
arr_logits = np.zeros(shape=(Ys[trained_p].shape[0], 3))
a = 1
for a in attempts:
    epoch = client.execute(f"SELECT MAX(Epoch) FROM {database}.{histores} WHERE ({database}.{histores}.id in ("
                f"SELECT id FROM {database}.{mapper} WHERE (ModelID = '{ModelID}') AND Profile='{trained_p}' AND Attempt={a}"
                ") );")[0][0]

    #! For tests use arr_logits[:-400,a-1] - hell knows why
    res= client.execute(
        query=f"SELECT Logits FROM {database}.logits_{l_type} WHERE ({database}.logits_{l_type}.id in ("
            f"SELECT id FROM {database}.{mapper} WHERE (ModelID = '{ModelID}') AND Profile='{trained_p}' AND Attempt={a}"
            f") AND Epoch = {epoch});" # Max(Epoch) for an attempt
            )[0][0]
    arr_logits[:len(res),a-1] = res
    
    errors.append(mean_absolute_error(
        y_true=Ys[trained_p], y_pred=arr_logits[:,a-1]
    ).numpy())
df_errors = pd.DataFrame(data=errors, columns=['mae'])
print(df_errors.sort_values('mae'))
indexes = df_errors[
    df_errors['mae'] < np.mean(df_errors['mae'])
    ].index

average = arr_logits[:,indexes].mean(axis=1)
MAE.update_state(y_true=Ys[trained_p],
                    y_pred=average)
RMSE.update_state(y_true=Ys[trained_p],
                    y_pred=average)
RS.update_state(y_true=np.expand_dims(Ys[trained_p],axis=1),
                y_pred=np.expand_dims(average,axis=1))
mae = MAE.result().numpy()*100
rmse = RMSE.result().numpy()*100
rs = RS.result().numpy()*100

RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
            Ys[trained_p]-average)))

if (l_type == 'test'):
    model_type = 'Testing'
else:
    model_type = 'Validation'

predicting_plot(profile=trained_p, file_name=f'Model № {ModelID}',
                model_loc='Modds/tmp/',
                model_type=model_type,
                iEpoch=f'{ModelID}-{l_type}',
                Y=Ys[trained_p],
                PRED=average,
                RMS=RMS,
                val_perf=[0, mae, rmse, rs],
                TAIL=average.shape[0],
                save_plot=True)

MAE.reset_states()
RMSE.reset_states()
RS.reset_states()
# %%