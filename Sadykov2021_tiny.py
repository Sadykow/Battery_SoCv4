#!/usr/bin/python
# %% [markdown]
# # Auto-regression implementation (Forward-Teaching)
import datetime
import logging
import os, sys, getopt    # OS, SYS, argc functions
from sys import platform  # Get type of OS

import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # File read
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import trange

from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
from py_modules.AutoFeedBack_GRU import AutoFeedBack
from py_modules.RobustAdam import RobustAdam
from cy_modules.utils import str2bool
from py_modules.plotting import predicting_plot
# %%
# Extract params
# try:
#     opts, args = getopt.getopt(sys.argv[1:],"hd:e:g:p:s:",
#                     ["help", "debug=", "epochs=",
#                      "gpu=", "profile="])
# except getopt.error as err: 
#     # output error, and return with an error code 
#     print (str(err)) 
#     print ('EXEPTION: Arguments requied!')
#     sys.exit(2)

opts = [('-d', 'False'), ('-e', '5'), ('-g', '0'), ('-p', 'FUDS'), ('-s', '40')]
mEpoch    : int = 10
GPU       : int = 0
profile   : str = 'DST'
out_steps : int = 10
for opt, arg in opts:
    if opt == '-h':
        print('HELP: Use following default example.')
        print('python *.py --debug False --epochs 50 --gpu 0 --profile DST')
        print('TODO: Create a proper help')
        sys.exit()
    elif opt in ("-d", "--debug"):
        if(str2bool(arg)):
            logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s --> %(levelname)s:%(message)s')
            logging.warning("Logger DEBUG")
        else:
            logging.basicConfig(level=logging.CRITICAL)
            logging.warning("Logger Critical")
    elif opt in ("-e", "--epochs"):
        mEpoch = int(arg)
    elif opt in ("-g", "--gpu"):
        GPU = int(arg)
    elif opt in ("-p", "--profile"):
        profile = (arg)
    elif opt in ("-s", "--steps"):
        out_steps = int(arg)
# %%
# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['font.family'] = 'Bender'

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
#! For numeric stability, set the default floating-point dtype to float32
tf.keras.backend.set_floatx('float32')
# %%
#! Check OS to change SymLink usage
if(platform=='win32'):
    Data    : str = 'DataWin\\'
else:
    Data    : str = 'Data/'
dataGenerator = DataGenerator(train_dir=f'{Data}A123_Matt_Set_2nd',
                              valid_dir=f'{Data}A123_Matt_Val_2nd',
                              test_dir=f'{Data}A123_Matt_Test',
                              columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                                'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                                ],
                              PROFILE_range = profile)
# %%
window = WindowGenerator(Data=dataGenerator,
                        input_width=500, label_width=out_steps, shift=1,
                        input_columns=['Current(A)', 'Voltage(V)',
                                                'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,
                        includeTarget=True, normaliseLabal=False,
                        shuffleTraining=False)
ds_train, xx_train, _ = window.train
ds_valid, xx_valid, _ = window.valid
#! Time to start rounding the charge to 2 or 3 decimals
xx_train[:,:,3] = np.round(xx_train[:,:,3], decimals=2)
xx_valid[:,:,3] = np.round(xx_valid[:,:,3], decimals=2)
yy_train : np.ndarray = np.asarray(list(ds_train.map(
                                lambda _, y: y[0,:,0]
                              ).as_numpy_iterator()
                          ))
yy_valid : np.ndarray = np.asarray(list(ds_valid.map(
                                lambda _, y: y[0,:,0]
                              ).as_numpy_iterator()
                          ))
yy_train = np.round(yy_train, decimals=2)
yy_valid = np.round(yy_valid, decimals=2)
# Entire Training set 
x_train = np.array(xx_train, copy=True, dtype=np.float32)
y_train = np.array(yy_train, copy=True, dtype=np.float32)

# For validation use same training
x_valid = np.array(xx_train[16800:25000,:,:], copy=True, dtype=np.float32)
y_valid = np.array(yy_train[16800:25000,:]  , copy=True, dtype=np.float32)
# x_valid = np.array(xx_train[:,:,:], copy=True, dtype=np.float32)
# y_valid = np.array(yy_train[:,:]  , copy=True, dtype=np.float32)

# For test dataset take the remaining profiles.
mid = int(xx_valid.shape[0]/2)+350
x_test_one = np.array(xx_valid[:mid,:,:], copy=True, dtype=np.float32)
y_test_one = np.array(yy_valid[:mid,:], copy=True, dtype=np.float32)
x_test_two = np.array(xx_valid[mid:,:,:], copy=True, dtype=np.float32)
y_test_two = np.array(yy_valid[mid:,:], copy=True, dtype=np.float32)
# %%
file_name : str = os.path.basename(__file__)[:-3]
model_loc : str = f'Models/{file_name}-{out_steps}steps/{profile}-models/'
iEpoch = 0
firstLog  : bool = True
try:
    for _, _, files in os.walk(model_loc):
        for file in files:
            if file.endswith('.ch'):
                iEpoch = int(os.path.splitext(file)[0])
    lstm_model : AutoFeedBack = AutoFeedBack(units=64,
            out_steps=out_steps, num_features=1
        )
    lstm_model.load_weights(f'{model_loc}{iEpoch}/{iEpoch}')
    firstLog = False
    print("Model Identefied. Continue training.")
    lstm_model : AutoFeedBack = tf.keras.models.load_model(
            filepath=f'{model_loc}/test',
            custom_objects={"AutoFeedBack": AutoFeedBack},
            compile=True
        )
except:
    print("Model Not Found, with some TF error.\n")
    lstm_model : AutoFeedBack = AutoFeedBack(units=64,
            out_steps=out_steps, num_features=1
        )
    firstLog = True

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        min_delta=0, patience=2, verbose=0, mode='min',
        baseline=None, restore_best_weights=False
    )
nanTerminate = tf.keras.callbacks.TerminateOnNaN()

# %%
#! I changed learning rate from 0.001 to 0.0001 after first run. If further fails
#!replace back. The drop was present.
lstm_model.compile(loss=tf.losses.MeanAbsoluteError(),
            optimizer=tf.optimizers.Adam(learning_rate = 0.0001), #!Start: 0.001
            metrics=[tf.metrics.MeanAbsoluteError(),
                     tf.metrics.RootMeanSquaredError(),
                     tfa.metrics.RSquare(y_shape=(out_steps,), dtype=tf.float32)
                    ])
while iEpoch < mEpoch:
    iEpoch+=1
    train_hist = lstm_model.fit(
                    x=x_train, y=y_train, epochs=1,
                    callbacks=[nanTerminate],
                    batch_size=1, shuffle=True
                )
    lstm_model.save(filepath=f'{model_loc}{iEpoch}',
                overwrite=True, include_optimizer=True,
                save_format='tf', signatures=None, options=None,
                save_traces=False
        )
    # Saving model
    lstm_model.save_weights(filepath=f'{model_loc}{iEpoch}/{iEpoch}',
            overwrite=True, save_format='tf', options=None
        )

    if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
        os.remove(f'{model_loc}{iEpoch-1}.ch')
    os.mknod(f'{model_loc}{iEpoch}.ch')

    # PERF = valid_step(x_valid, y_valid)
    # hist_df = pd.DataFrame(data={
    #         'loss'   : [np.array(loss_value)],
    #         'mae'    : [np.array(MAE.result())],
    #         'rmse'   : [np.array(RMSE.result())],
    #         'rsquare': [np.array(RSquare.result())]
    #     })
    # hist_df['vall_loss'] = PERF[0]
    # hist_df['val_mean_absolute_error'] = PERF[1]
    # hist_df['val_root_mean_squared_error'] = PERF[2]
    # hist_df['val_r_square'] = PERF[3]
    
    # or save to csv:
    hist_df = pd.DataFrame(train_hist.history)
    PERF = lstm_model.evaluate(x=x_valid,
                               y=y_valid,
                               batch_size=1,
                               verbose=1)
    hist_df['vall_loss'] = PERF[0]
    hist_df['val_mean_absolute_error'] = PERF[1]
    hist_df['val_root_mean_squared_error'] = PERF[2]
    hist_df['val_r_square'] = PERF[3]

    with open(f'{model_loc}history-{profile}.csv', mode='a') as f:
        if(firstLog):
            hist_df.to_csv(f, index=False)
            firstLog = False
        else:
            hist_df.to_csv(f, index=False, header=False)
    
    # PRED = PERF[4]
    # RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
    #             y_valid[::,]-PRED)))
    # # otherwise the right y-label is slightly clipped
    # predicting_plot(profile=profile, file_name='myModel №1',
    #                 model_loc=model_loc,
    #                 model_type='LST Test - Train dataset',
    #                 iEpoch=f'val-{iEpoch}',
    #                 Y=y_valid,
    #                 PRED=PRED,
    #                 RMS=RMS,
    #                 val_perf=PERF[:4],
    #                 TAIL=y_valid.shape[0],
    #                 save_plot=True,
    #                 RMS_plot=False) #! Saving memory from high errors.
    # if(PERF[-3] <=0.024): # Check thr RMSE
    #     print("RMS droped around 2.4%. Breaking the training")
    #     break
    VIT_input = x_valid[0,:,:3]
    SOC_input = x_valid[0,:,3:]
    PRED = np.zeros(shape=(y_valid.shape[0],), dtype=np.float32)
    for i in trange(y_valid.shape[0]):
        logits = lstm_model.predict(
                                x=np.expand_dims(
                                    np.concatenate(
                                        (VIT_input, SOC_input),
                                        axis=1),
                                    axis=0),
                                batch_size=1
                            )
        VIT_input = x_valid[i,:,:3]
        SOC_input = np.concatenate(
                            (SOC_input, np.expand_dims(logits,axis=0)),
                            axis=0)[1:,:]
        PRED[i] = logits
    MAE = np.mean(tf.keras.backend.abs(y_valid[::,-1]-PRED))
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_valid[::,-1]-PRED)))
    # Time range
    test_time = range(0,PRED.shape[0])
    def format_func(value, _):
        return int(value*100)

    fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
    ax1.plot(test_time, y_valid[:,-1], label="Actual")
    ax1.plot(test_time, PRED, label="Prediction")
    ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
    ax1.set_xlabel("Time Slice (s)", fontsize=32)
    ax1.set_ylabel("SoC (%)", fontsize=32)
    #if RMS_plot:
    ax2 = ax1.twinx()
    ax2.plot(test_time,
        RMS,
        label="RMS error", color='#698856')
    ax2.fill_between(test_time,
        RMS,
        color='#698856')
    ax2.set_ylabel('Error', fontsize=32, color='#698856')
    ax2.tick_params(axis='y', labelcolor='#698856', labelsize=24)
    ax2.set_ylim([-0.1,1.6])
    ax1.set_title(
        #f"{file_name} {model_type}. {profile}-trained",
        f"VITpSoC №1 Train Dataset. {profile}-trained. {out_steps}-steps",
        fontsize=36)
    ax1.legend(prop={'size': 32})
    ax1.tick_params(axis='both', labelsize=24)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax1.set_ylim([-0.1,1.2])
    fig.tight_layout()
    textstr = '\n'.join((
        '$MAE  = {0:.2f}%$'.format(np.mean(MAE)*100, ),
        '$RMSE = {0:.2f}%$'.format(np.mean(RMS)*100, )
        # '$R2  = nn.nn%$'
            ))
    ax1.text(0.65, 0.80, textstr, transform=ax1.transAxes, fontsize=30,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.savefig(f'{model_loc}{profile}-{iEpoch}.svg')
    fig.clf()
    plt.close()
# %% [Running performance tests]
test_S = 3000
test_N = 8000
x_test = x_train[test_S:test_S+test_N,:,:]
y_test = y_train[test_S:test_S+test_N,:]
# plt.plot(y_test[:,0])
VIT_input = x_test[0,:,:3]
SOC_input = x_test[0,:,3:]
# VIT_input = np.ones(shape=x_test[0,:,:3].shape)*x_test[-1,0,:3]
# SOC_input = np.ones(shape=x_test[0,:,3:].shape)*0.60#x_test[-1,,3:]
PRED = np.zeros(shape=(y_test.shape[0],), dtype=np.float32)
for i in trange(y_test.shape[0]):
    logits = lstm_model.predict(
                            x=np.expand_dims(
                                np.concatenate(
                                    (VIT_input, SOC_input),
                                    axis=1),
                                axis=0),
                            batch_size=1
                        )
    VIT_input = x_test[i,:,:3]
    SOC_input = np.concatenate(
                        (SOC_input, np.expand_dims(logits,axis=0)),
                        axis=0)[1:,:]
    PRED[i] = logits
MAE = np.mean(tf.keras.backend.abs(y_test[::,-1]-PRED))
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
            y_test[::,-1]-PRED)))
test_time = range(0,PRED.shape[0])
def format_func(value, _):
    return int(value*100)
# %%
fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
ax1.plot(test_time, y_test[:,-1], label="Actual")
ax1.plot(test_time, PRED, label="Prediction")
ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
ax1.set_xlabel("Time Slice (s)", fontsize=32)
ax1.set_ylabel("SoC (%)", fontsize=32)
#if RMS_plot:
ax2 = ax1.twinx()
ax2.plot(test_time,
    RMS,
    label="RMS error", color='#698856')
ax2.fill_between(test_time,
    RMS,
    color='#698856')
ax2.set_ylabel('Error', fontsize=32, color='#698856')
ax2.tick_params(axis='y', labelcolor='#698856', labelsize=24)
ax2.set_ylim([-0.1,1.6])
ax1.set_title(
    #f"{file_name} {model_type}. {profile}-trained",
    f"Initial Discharging State. Constant 71% charge",
    fontsize=36)
ax1.legend(prop={'size': 32})
ax2.legend(loc='center right', bbox_to_anchor=(1.0,0.80), prop={'size': 32})
ax1.tick_params(axis='both', labelsize=24)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax1.set_ylim([-0.1,1.2])
fig.tight_layout()
textstr = '\n'.join((
    '$MAE  = {0:.2f}%$'.format(np.mean(MAE)*100, ),
    '$RMSE = {0:.2f}%$'.format(np.mean(RMS)*100, )
    # '$R2  = nn.nn%$'
        ))
ax1.text(0.66, 0.74, textstr, transform=ax1.transAxes, fontsize=30,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
# fig.savefig(f'{model_loc}images/train-i71percent.svg')
# %%
VIT_input = x_test_one[0,:,:3]
SOC_input = x_test_one[0,:,3:]
PRED = np.zeros(shape=(y_test_one.shape[0],), dtype=np.float32)
for i in trange(y_test_one.shape[0]):
    logits = lstm_model.predict(
                            x=np.expand_dims(
                                np.concatenate(
                                    (VIT_input, SOC_input),
                                    axis=1),
                                axis=0),
                            batch_size=1
                        )
    VIT_input = x_test_one[i,:,:3]
    SOC_input = np.concatenate(
                        (SOC_input, np.expand_dims(logits,axis=0)),
                        axis=0)[1:,:]
    PRED[i] = logits
MAE = np.mean(tf.keras.backend.abs(y_test_one[::,-1]-PRED))
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
            y_test_one[::,-1]-PRED)))

# Time range
test_time = range(0,PRED.shape[0])
def format_func(value, _):
    return int(value*100)

fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
ax1.plot(test_time, y_test_one[:,-1], label="Actual")
ax1.plot(test_time, PRED, label="Prediction")
ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
ax1.set_xlabel("Time Slice (s)", fontsize=32)
ax1.set_ylabel("SoC (%)", fontsize=32)
#if RMS_plot:
ax2 = ax1.twinx()
ax2.plot(test_time,
    RMS,
    label="RMS error", color='#698856')
ax2.fill_between(test_time,
    RMS,
    color='#698856')
ax2.set_ylabel('Error', fontsize=32, color='#698856')
ax2.tick_params(axis='y', labelcolor='#698856', labelsize=24)
ax2.set_ylim([-0.1,1.6])
ax1.set_title(
    #f"{file_name} {model_type}. {profile}-trained",
    f"VITpSoC №1 Test. {profile}-trained over DST set. {out_steps}-steps",
    fontsize=36)
ax1.legend(prop={'size': 32})
ax2.legend(loc='center right', bbox_to_anchor=(1.0,0.80), prop={'size': 32})
ax1.tick_params(axis='both', labelsize=24)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax1.set_ylim([-0.1,1.2])
fig.tight_layout()
textstr = '\n'.join((
    '$MAE  = {0:.2f}%$'.format(np.mean(MAE)*100, ),
    '$RMSE = {0:.2f}%$'.format(np.mean(RMS)*100, )
    # '$R2  = nn.nn%$'
        ))
ax1.text(0.66, 0.74, textstr, transform=ax1.transAxes, fontsize=30,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
fig.savefig(f'{model_loc}images/test_oneFull2.svg')
# %%
VIT_input = x_test_two[0,:,:3]
SOC_input = x_test_two[0,:,3:]
PRED = np.zeros(shape=(y_test_two.shape[0],), dtype=np.float32)
for i in trange(y_test_two.shape[0]):
    logits = lstm_model.predict(
                            x=np.expand_dims(
                                np.concatenate(
                                    (VIT_input, SOC_input),
                                    axis=1),
                                axis=0),
                            batch_size=1
                        )
    VIT_input = x_test_two[i,:,:3]
    SOC_input = np.concatenate(
                        (SOC_input, np.expand_dims(logits,axis=0)),
                        axis=0)[1:,:]
    PRED[i] = logits
MAE = np.mean(tf.keras.backend.abs(y_test_two[::,-1]-PRED))
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
            y_test_two[::,-1]-PRED)))

# Time range
test_time = range(0,PRED.shape[0])
def format_func(value, _):
    return int(value*100)

fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
ax1.plot(test_time, y_test_two[:,-1], label="Actual")
ax1.plot(test_time, PRED, label="Prediction")
ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
ax1.set_xlabel("Time Slice (s)", fontsize=32)
ax1.set_ylabel("SoC (%)", fontsize=32)
#if RMS_plot:
ax2 = ax1.twinx()
ax2.plot(test_time,
    RMS,
    label="RMS error", color='#698856')
ax2.fill_between(test_time,
    RMS,
    color='#698856')
ax2.set_ylabel('Error', fontsize=32, color='#698856')
ax2.tick_params(axis='y', labelcolor='#698856', labelsize=24)
ax2.set_ylim([-0.1,1.6])
ax1.set_title(
    #f"{file_name} {model_type}. {profile}-trained",
    f"VITpSoC №1 Train Dataset. {profile}-trained over US06 set. {out_steps}-steps",
    fontsize=36)
ax1.legend(prop={'size': 32})
ax2.legend(loc='center right', bbox_to_anchor=(1.0,0.80), prop={'size': 32})
ax1.tick_params(axis='both', labelsize=24)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax1.set_ylim([-0.1,1.2])
fig.tight_layout()
textstr = '\n'.join((
    '$MAE  = {0:.2f}%$'.format(np.mean(MAE)*100, ),
    '$RMSE = {0:.2f}%$'.format(np.mean(RMS)*100, )
    # '$R2  = nn.nn%$'
        ))
ax1.text(0.66, 0.74, textstr, transform=ax1.transAxes, fontsize=30,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
fig.savefig(f'{model_loc}images/test_twoFull2.svg')
# %%
# for j in range(2,6):
#     lstm_model.load_weights(f'{model_loc}{j}/variables/variables')
#     VIT_input = x_valid[0,:,:3]
#     SOC_input = x_valid[0,:,3:]
#     PRED = np.zeros(shape=(y_valid.shape[0],), dtype=np.float32)
#     for i in trange(y_valid.shape[0]):
#         logits = lstm_model.predict(
#                                 x=np.expand_dims(
#                                     np.concatenate(
#                                         (VIT_input, SOC_input),
#                                         axis=1),
#                                     axis=0),
#                                 batch_size=1
#                             )
#         VIT_input = x_valid[i,:,:3]
#         SOC_input = np.concatenate(
#                             (SOC_input, np.expand_dims(logits,axis=0)),
#                             axis=0)[1:,:]
#         PRED[i] = logits
#     plt.plot(y_valid[:,0])
#     plt.plot(PRED)
#     plt.show()
# %%
# optimiser = tf.optimizers.Adam(learning_rate = 0.0001)
# loss_fn   = tf.losses.MeanAbsoluteError()
# MAE     = tf.metrics.MeanAbsoluteError()
# RMSE    = tf.metrics.RootMeanSquaredError()
# # %%
# sh_i = np.arange(y_train.shape[0])
# for i in sh_i[:1]:
#     with tf.GradientTape() as tape:
#         logits     = lstm_model(x_train[i:i+1,:,:], training=True)
#         loss_value = loss_fn(y_train[i:i+1,:], logits)
#     grads = tape.gradient(loss_value, lstm_model.trainable_weights)
#     # optimiser.update_loss(prev_loss, loss_value)
#     optimiser.apply_gradients(zip(grads, lstm_model.trainable_weights))
#     #MAE.update_state(y_true=y[:1], y_pred=logits)
#     #RMSE.update_state(y_true=y[:], y_pred=logits)

# %%
# optimiser = tf.optimizers.Adam(learning_rate = 0.001)
# loss_fn   = tf.losses.MeanAbsoluteError()
# @tf.function
# def train_single_st(x, y, prev_loss):
#     with tf.GradientTape() as tape:
#         logits     = lstm_model(x, training=True)
#         loss_value = loss_fn(y, logits)
#     grads = tape.gradient(loss_value, lstm_model.trainable_weights)
#     # optimiser.update_loss(prev_loss, loss_value)
#     optimiser.apply_gradients(zip(grads, lstm_model.trainable_weights))
#     MAE.update_state(y_true=y[:1], y_pred=logits)
#     RMSE.update_state(y_true=y[:], y_pred=logits)
#     # RSquare.update_state(y_true=y[:], y_pred=logits)
#     return loss_value

# @tf.function
# def test_step(x):
#     return lstm_model(x, training=False)

# def valid_step(x, y):
#     logits  = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
#     loss    = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
#     mae     = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
#     rmse    = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
#     rsquare = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
#     for i in trange(y.shape[0]):
#         logits[i] = test_step(x[i:i+1,:,:])
#         MAE.update_state(y_true=y[i:i+1], y_pred=logits[i])
#         RMSE.update_state(y_true=y[i:i+1], y_pred=logits[i])
#         # RSquare.update_state(y_true=y[i:i+1], y_pred=logits[i])
#         loss[i]    = loss_fn(y[i:i+1], logits[i])
#         mae[i]     = MAE.result()
#         rmse[i]    = RMSE.result()
#         # rsquare[i] = RSquare.result()
#     return [np.mean(loss), np.mean(mae), np.mean(rmse), np.mean(rsquare), logits]

# MAE     = tf.metrics.MeanAbsoluteError()
# RMSE    = tf.metrics.RootMeanSquaredError()
# # RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)
# pbar = tqdm(total=y_train.shape[0])
# sh_i = np.arange(y_train.shape[0])
# # np.random.shuffle(sh_i)
# for i in sh_i[:]:
#     loss_value = train_single_st(x_train[i:i+1,:,:], y_train[i:i+1,:],
#                                 None)
#     # Progress Bar
#     pbar.update(1)
#     pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
#                         f'loss: {loss_value:.4f} - '
#                         f'mae: {MAE.result():.4f} - '
#                         f'rmse: {RMSE.result():.4f} - '
#                         # f'rsquare: {RSquare.result():.4f}'
#                         )
# pbar.close()
# lstm_model.reset_metrics()
# lstm_model.reset_states()
# logits  = np.zeros(shape=(y_valid.shape[0], ), dtype=np.float32)
# for i in trange(y_valid.shape[0]):
#     logits[i] = test_step(x_valid[i:i+1,:,:])[-1]

# %%
# lstm_model.compile(loss=tf.losses.MeanAbsoluteError(),
#             optimizer=tf.optimizers.Adam(learning_rate = 0.0001),
#             metrics=[tf.metrics.MeanAbsoluteError(),
#                      tf.metrics.RootMeanSquaredError(),
#                     #  tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)
#                     ])
# while iEpoch < mEpoch:
#     iEpoch+=1
#     print(f"Epoch {iEpoch}/{mEpoch}")
    
#     train_hist = lstm_model.fit(
#                     x=x_train, y=y_train, epochs=1,
#                     validation_data=(x_valid, y_valid),
#                     callbacks=[early_stopping, nanTerminate],
#                     batch_size=1, shuffle=False
#                     #run_eagerly=True
#                 )
#     lstm_model.save(filepath=f'{model_loc}{iEpoch}',
#                         overwrite=True, include_optimizer=True,
#                         save_format='tf', signatures=None, options=None,
#                         save_traces=True
#                 )
#     if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
#         os.remove(f'{model_loc}{iEpoch-1}.ch')
#     os.mknod(f'{model_loc}{iEpoch}.ch')

#     # Saving history variable
#     # convert the history.history dict to a pandas DataFrame:     
#     hist_df = pd.DataFrame(train_hist.history)
#     # or save to csv:
#     with open(f'{model_loc}history-{profile}.csv', mode='a') as f:
#         if(firstLog):
#             hist_df.to_csv(f, index=False)
#             firstLog = False
#         else:
#             hist_df.to_csv(f, index=False, header=False)
    
#     #! Run the Evaluate function
#     #! Replace with tf.metric function.
#     PRED = lstm_model.predict(x_valid, batch_size=1)
#     RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
#                 y_valid[::,]-PRED)))
#     PERF = lstm_model.evaluate(x=x_valid,
#                                y=y_valid,
#                                batch_size=1,
#                                verbose=0)
#     # otherwise the right y-label is slightly clipped
#     predicting_plot(profile=profile, file_name='myModel №2',
#                     model_loc=model_loc,
#                     model_type='LSTM Test - Train dataset',
#                     iEpoch=f'val-{iEpoch}',
#                     Y=y_valid,
#                     PRED=PRED,
#                     RMS=RMS,
#                     val_perf=PERF,
#                     TAIL=y_valid.shape[0],
#                     save_plot=True)
#     if(PERF[-2] <=0.024): # Check thr RMSE
#         print("RMS droped around 2.4%. Breaking the training")
#         break
# %%
# # %%
# lstm_model.build((1,500,4))
# lstm_model.save(filepath=f'{model_loc}test/12-2',
#                     overwrite=True, include_optimizer=True,
#                     save_format='tf', signatures=None, options=None,
#                     save_traces=True
#             )
# # Convert the model to Tensorflow Lite and save.
with open(f'{model_loc}myModel-№2-{profile}.tflite', 'wb') as f:
    f.write(
        tf.lite.TFLiteConverter.from_saved_model(
                saved_model_dir=f'{model_loc}test/12-2'
            ).convert()
        )
        tf.lite.TFLiteConverter.from_keras_model(
                model=lstm_model
            ).convert()
        )