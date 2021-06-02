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

sys.path.append(os.getcwd() + '/..')
from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
from py_modules.AutoFeedBack import AutoFeedBack
from py_modules.RobustAdam import RobustAdam
from cy_modules.utils import str2bool
from py_modules.plotting import predicting_plot

opts = [('-d', 'False'), ('-e', '5'), ('-g', '0'), ('-p', 'FUDS'), ('-s', '30')]
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
    Data    : str = '../DataWin\\'
else:
    Data    : str = '../Data/'
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
model_loc : str = f'../Models/Sadykov2021-{out_steps}steps/{profile}-models/'
nEpoch = 20
lstm_model : AutoFeedBack = AutoFeedBack(units=510,
            out_steps=out_steps, num_features=1
        )
def format_func(value, _):
    return int(value*100)
# %%
for iEpoch in range(20, 21):
    lstm_model.load_weights(f'{model_loc}{iEpoch}/{iEpoch}')
    lstm_model.compile(loss=tf.losses.MeanAbsoluteError(),
            optimizer=tf.optimizers.Adam(learning_rate = 0.0001), #!Start: 0.001
            metrics=[tf.metrics.MeanAbsoluteError(),
                     tf.metrics.RootMeanSquaredError(),
                     tfa.metrics.RSquare(y_shape=(out_steps,), dtype=tf.float32)
                    ])    
    fig, axs = plt.subplots(1, 3, figsize=(42,12), dpi=600)
    #! First validate on training set:
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

    ax1 = axs[0]
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
        f"Result of the training on FUDS-{iEpoch}",
        fontsize=36)
    ax1.legend(prop={'size': 32})
    ax1.tick_params(axis='both', labelsize=24)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax1.set_ylim([-0.1,1.2])
    textstr = '\n'.join((
        '$MAE  = {0:.2f}%$'.format(np.mean(MAE)*100, ),
        '$RMSE = {0:.2f}%$'.format(np.mean(RMS)*100, )
        # '$R2  = nn.nn%$'
            ))
    ax1.text(0.65, 0.80, textstr, transform=ax1.transAxes, fontsize=30,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    #! Try on DST dataset
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
    ax1 = axs[1]
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
        f"Result of the validation on DST-{iEpoch}",
        fontsize=36)
    ax1.legend(prop={'size': 32})
    ax2.legend(loc='center right', bbox_to_anchor=(1.0,0.80), prop={'size': 32})
    ax1.tick_params(axis='both', labelsize=24)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax1.set_ylim([-0.1,1.2])
    # fig.tight_layout()
    textstr = '\n'.join((
        '$MAE  = {0:.2f}%$'.format(np.mean(MAE)*100, ),
        '$RMSE = {0:.2f}%$'.format(np.mean(RMS)*100, )
        # '$R2  = nn.nn%$'
            ))
    ax1.text(0.66, 0.74, textstr, transform=ax1.transAxes, fontsize=30,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # fig.savefig(f'{model_loc}images/test_oneFull2.svg')
    #! Validate the next one
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

    ax1 = axs[2]
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
        f"Result of the validation on US06-{iEpoch}",
        fontsize=36)
    ax1.legend(prop={'size': 32})
    ax2.legend(loc='center right', bbox_to_anchor=(1.0,0.80), prop={'size': 32})
    ax1.tick_params(axis='both', labelsize=24)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax1.set_ylim([-0.1,1.2])
    textstr = '\n'.join((
        '$MAE  = {0:.2f}%$'.format(np.mean(MAE)*100, ),
        '$RMSE = {0:.2f}%$'.format(np.mean(RMS)*100, )
        # '$R2  = nn.nn%$'
            ))
    ax1.text(0.66, 0.74, textstr, transform=ax1.transAxes, fontsize=30,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.tight_layout()
    fig.savefig(f'figures/sadykov/{iEpoch}.svg')
    fig.clf()
    plt.close()
# %%