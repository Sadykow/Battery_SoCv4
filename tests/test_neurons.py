# %% [markdown]
# Scrpit intended to test different neurons to find the most optimal for this
# case
# %%
import datetime
from functools import reduce
import logging
import os, sys, getopt    # OS, SYS, argc functions
from sys import platform  # Get type of OS

import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # File read
import tensorflow as tf  # Tensorflow and Numpy replacement
import tensorflow_addons as tfa
from tqdm import tqdm, trange

sys.path.append(os.getcwd() + '/..')
from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator

import sys, time
if (sys.version_info[1] < 9):
    LIST = list
    from typing import List as list
    from typing import Tuple as tuple

# Formula to get nodes
# N_h = (N_s) / (a(Ni+No))
# Varying alpha
def round_down(x : float, factor : int = 10) -> int:
    """ Round up to a factor. Uses it to create hidden neurons, or Buffer size.
    TODO: Make it a smarter rounder.
    Args:
        x (float): Original float value.
        factor (float): Factor towards which it has to be rounder

    Returns:
        int: Rounded up value based on factor.
    """
    if(factor == 10):
        return int(np.floor(x / 10)) * 10
    elif(factor == 100):
        return int(np.floor(x / 100)) * 100
    elif(factor == 1000):
        return int(np.floor(x / 1000)) * 1000
    else:
        print("Factor of {} not implemented.".format(factor))
        return None
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
#! Select GPU for usage. CPU versions ignores it.
#!! Learn to check if GPU is occupied or not.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    #! With /device/GPU:1 the output was faster.
    #! need to research more why.
    # tf.config.experimental.set_visible_devices(
    #                         physical_devices[GPU], 'GPU')

    # if GPU == 1:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(
                            device=device, enable=True)
    logging.info("GPU found and memory growth enabled") 
    
    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    logging.info("GPU found") 
    logging.debug(f"\nPhysical GPUs: {len(physical_devices)}"
                  f"\nLogical GPUs: {len(logical_devices)}")
#! For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float32')
# %%
Data    : str = '../Data/'
look_back : int = 500
dataGenerator = DataGenerator(train_dir=f'{Data}A123_Matt_Set_2nd',
                              valid_dir=f'{Data}A123_Matt_Val_2nd',
                              test_dir=f'{Data}A123_Matt_Test',
                              columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                                'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                                ],
                              PROFILE_range = 'FUDS')
window = WindowGenerator(Data=dataGenerator,
                        input_width=look_back, label_width=1, shift=0,
                        input_columns=['Current(A)', 'Voltage(V)',
                                                'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,
                        includeTarget=False, normaliseLabal=False,
                        shuffleTraining=False)
ds_train, x_train, y_train = window.train
# ds_valid, x_valid, y_valid = window.valid

# For validation use same training
#! See if you can use validation over another battery
x_valid = np.array(x_train[17000:,:,:,:], copy=True, dtype=np.float32)
y_valid = np.array(y_train[17000:,:]  , copy=True, dtype=np.float32)

x_train = np.array(x_train[:17000,:,:,:], copy=True, dtype=np.float32)
y_train = np.array(y_train[:17000,:]  , copy=True, dtype=np.float32)
# %%
def scheduler(_ : int, lr : float) -> float:
    """ Scheduler
    round(model.optimizer.lr.numpy(), 5)

    Args:
        epoch (int): [description]
        lr (float): [description]

    Returns:
        float: [description]
    """
    #! Think of the better sheduler
    if (epoch < 2):
        return iLr
    else:
        # lr = tf_round(x=lr * tf.math.exp(-0.2), decimals=5)
        lr = lr * tf.math.exp(-0.25)
        if lr >= 0.0002:
            return lr
        else:
            return  0.00005

# (41068, 1, 500, 3) - Training
# (8200, 1, 500, 3) - Validation
# 41068 / (0.16*(500+1))
alpha   : int = 10 # / 2 >> 0.16 #!ROUND-DOWN
iLr     : float = 0.001
h_nodes : int = np.floor(
                41068 /\
        (alpha * (look_back +1) )
        )
alpha_tic : float = time.perf_counter()
while alpha >= 0.15:
    h_nodes : int = int(np.floor(
                41068 /\
        (alpha * (look_back +1) )
        ))
    print(f"Alpha: {alpha} -- The number of hidden nodes: {h_nodes}.")
    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.InputLayer(input_shape=(x_train.shape[-2:]),
                                batch_size=1),
        tf.keras.layers.LSTM(
            units=int(h_nodes/3), activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            unit_forget_bias=True, kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.2,
            recurrent_dropout=0.0, implementation=2, return_sequences=True, #!
            return_state=False, go_backwards=False, stateful=False,
            time_major=False, unroll=False#, batch_input_shape=(None, 2, 500, 3)
        ),
        tf.keras.layers.LSTM(
            units=int(h_nodes/3), activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            unit_forget_bias=True, kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.2,
            recurrent_dropout=0.0, implementation=2, return_sequences=True, #!
            return_state=False, go_backwards=False, stateful=False,
            time_major=False, unroll=False#, batch_input_shape=(None, 2, 500, 3)
        ),
        tf.keras.layers.LSTM(
            units=int(h_nodes/3), activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            unit_forget_bias=True, kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.2,
            recurrent_dropout=0.0, implementation=2, return_sequences=False, #!
            return_state=False, go_backwards=False, stateful=False,
            time_major=False, unroll=False#, batch_input_shape=(None, 2, 500, 3)
        ),
        #tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1,
                            activation='sigmoid')
    ])
    model.compile(loss=tf.losses.MeanAbsoluteError(),
        optimizer=tf.optimizers.Adam(learning_rate=iLr,
                beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False),
        metrics=[tf.metrics.MeanAbsoluteError(),
                    tf.metrics.RootMeanSquaredError(),
                    tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
        #run_eagerly=True
    )
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    firstLog : bool = True
    for epoch in range(10):
        # Evaluation time for train set
        tic : float = time.perf_counter()
        history_log = model.fit(x=x_train[:,0,:,:], y=y_train[:,0,:],
                        epochs=1, batch_size=1,
                        callbacks = [], #! reduce_lr
                        shuffle=True, verbose = 0
                    )
        toc : float = time.perf_counter() - tic
        
        hist_df = pd.DataFrame(history_log.history)
        hist_df['Time(s)'] = toc
        hist_df['N_Params'] = model.count_params()
        model.save(filepath=f'Models/LSTM-{epoch}',
                    overwrite=True, include_optimizer=True,
                    save_format='h5', signatures=None, options=None,
                    save_traces=True
            )
        time.sleep(3)
        file_size : int = os.path.getsize(f'Models/LSTM-{epoch}')
        os.remove(f'Models/LSTM-{epoch}')
        print(
                f'Epoch - {epoch+1} - Time - {toc}\n'
                f'MAE {history_log.history["mean_absolute_error"][0]} -- ' 
                f'RMSE {history_log.history["root_mean_squared_error"][0]} -- '
            )
        # Evaluation time for test set
        tic = time.perf_counter()
        PERF = model.evaluate(x=x_valid[:,0,:,:],
                                y=y_valid[:,0,:],
                                batch_size=1,
                                verbose=0)
        val_toc : float = time.perf_counter() - tic

        hist_df[['val_loss', 'val_mae', 'val_rmse', 'val_r2']] = PERF
        hist_df['val_Time(s)'] = val_toc

        # Single sample time to predict
        tic = time.perf_counter()
        model(x_valid[0,:1,:,:], training=False)
        single_toc : float = time.perf_counter() - tic
        hist_df['single_Time(s)'] = single_toc
        hist_df['size'] = file_size

        # or save to csv:
        with open(f'Models/LSTM2-history-{h_nodes}.csv', mode='a') as f:
            if(firstLog):
                hist_df.to_csv(f, index=False)
                firstLog = False
            else:
                hist_df.to_csv(f, index=False, header=False)
        plt.plot(model.predict(x_valid[:,0,:,:], batch_size=1, verbose=1))
        plt.plot(y_valid[:,0,:])
        plt.show()
        print(
                f'Time - {val_toc} -- sTime: {single_toc} Size: {file_size}\n'
                f'val_MAE {PERF[1]} -- val_RMSE {PERF[2]} '
            )
    # Reduce Alpha
    alpha /= 2
    print('\n\n')
alpha_toc : float = time.perf_counter() - alpha_tic
print(f'Elapsed time for all alphas (m): {alpha_toc/60}')
# %%