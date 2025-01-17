#!/usr/bin/python
# %% [markdown]
# # # 1
# # #
# # LSTM for SoC by Ephrem Chemali 2017
# In this paper, we showcase how recurrent neural networks with an LSTM cell, a 
# machine learning technique, can accu-ately estimate SOC and can do so by
# self-learning the network parameters.

# Typical dataset used to train the networks is given by
#  D = {(Ψ 1 , SOC 1 ∗ ), (Ψ 2 , SOC 2 ∗ ), ..., (Ψ N , SOC N)}
# The vector of inputs is defined as 
# Ψ k = [V (k), I(k), T (k)], where V (k), I(k), T (k)

#? A final fully-connected layer performs a linear transformation on the 
#?hidden state tensor h k to obtain a single estimated SOC value at time step k.
#?This is done as follows: SOC k = V out h k + b y ,

#* Loss: sumN(0.5*(SoC-SoC*)^2)
#* Adam optimization: decay rates = b1=0.9 b2=0.999, alpha=10e-4, ?epsilon=10e-8

#* Drive cycles: over 100,000 time steps
#*  1 batch at a time.
#* Erors metricks: MAE, RMS, STDDEV, MAX error

# Data between 0 10 25C for training. Apperently mostly discharge cycle.
# Drive Cycles which included HWFET, UDDS, LA92 and US06
#* The network is trained on up to 8 mixed drive cycles while validation is
#*performed on 2 discharge test cases. In addition, a third test case, called
#*the Charging Test Case, which includes a charging profile is used to validate
#*the networks performance during charging scenarios.

#* LSTM-RNN with 500 nodes. Took 4 hours.
#* The MAE achieved on each of the first two test cases is 
#*0.807% and 1.252% (0.00807 and 0.01252)
#*15000 epoch.
#? This file used to train only on a FUDS dataset and validate against another
#?excluded FUDS datasets. In this case, out of 12 - 2 for validation.
#?Approximately 15~20% of provided data.
#? Testing performed on any datasets.
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

from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
from py_modules.utils import str2bool
from py_modules.plotting import predicting_plot
# %%
# Extract params
# try:
#     opts, args = getopt.getopt(sys.argv[1:],"hd:e:g:p:",
#                     ["help", "debug=", "epochs=",
#                      "gpu=", "profile="])
# except getopt.error as err: 
#     # output error, and return with an error code 
#     print (str(err)) 
#     print ('EXEPTION: Arguments requied!')
#     sys.exit(2)

opts = [('-d', 'False'), ('-e', '50'), ('-g', '0'), ('-p', 'FUDS')]
mEpoch  : int = 10
GPU     : int = 0
profile : str = 'DST'
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
        #! Another alternative is to use
        #!:$ export CUDA_VISIBLE_DEVICES=0,1 && python *.py
        GPU = int(arg)
    elif opt in ("-p", "--profile"):
        profile = (arg)
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
#! Select GPU for usage. CPU versions ignores it.
#!! Learn to check if GPU is occupied or not.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    #! With /device/GPU:1 the output was faster.
    #! need to research more why.
    # tf.config.experimental.set_visible_devices(
    #                         physical_devices[GPU], 'GPU')

    #if GPU == 1:
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
#! Check OS to change SymLink usage
if(platform=='win32'):
    Data    : str = 'DataWin\\'
else:
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
# file_name : str = os.path.basename(__file__)[:-3]
# model_loc : str = f'Models/{file_name}/{profile}-models/'
# N_seconds = 8000
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14,12), dpi=600)
# titles = [['Voltage', 'Current'],
#           ['Temperature', 'State of Charge']]
# units = [['Volts', 'Amps'],
#           ['Degrees', 'Percentage']]
# test_time = np.linspace(0, N_seconds/60, N_seconds)
# y_axis_data = [[dataGenerator.valid_df[:8000,1], dataGenerator.valid_df[:8000,0]],
#                [dataGenerator.valid_df[:8000,2], dataGenerator.valid_SoC[:8000,0]*100]]
# colors = [['#FF0000', '#0000FF'],
#           ['m','k']]
# fig.suptitle('Pre training input sample of single cell', fontsize=38)
# for (ax_row, titles_row,
#      units_row, y_row, colors_row) in zip(axs, titles,
#                               units, y_axis_data, colors):
#     for (ax, title,
#          unit, y, color) in zip(ax_row, titles_row,
#                          units_row, y_row, colors_row):
#         ax.plot(test_time, y, '-', color=color)
#         ax.set_title(f'{title} snapshot', fontsize=32)
#         ax.set_ylabel(f'{unit}', fontsize=32)
#         ax.set_xlabel('Time Slice (min)', fontsize=28)
#         ax.tick_params(axis='both', labelsize=22)
# fig.tight_layout()
# fig.savefig(f'{model_loc}pre-training-samples.svg')
# %%
window = WindowGenerator(Data=dataGenerator,
                        input_width=500, label_width=1, shift=0,
                        input_columns=['Current(A)', 'Voltage(V)',
                                                'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,#!See what you can do 
                        includeTarget=False, normaliseLabal=False,
                        shuffleTraining=False)
_, x_train, y_train = window.train
_, x_valid, y_valid = window.valid

# Wrap data in Dataset objects.
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

# Entire Training set 
# x_train = np.array(xx_train, copy=True, dtype=np.float32)
# y_train = np.array(yy_train, copy=True, dtype=np.float32)

# For validation use same training
#! See if you can use validation over another battery
x_valid = np.array(x_train[16800:25000,:,:], copy=True, dtype=np.float32)
y_valid = np.array(y_train[16800:25000,:]  , copy=True, dtype=np.float32)

# For test dataset take the remaining profiles.
#! See if you can use testing of same profile but another battery
mid = int(x_valid.shape[0]/2)+350
x_test_one = np.array(x_valid[:mid,:,:], copy=True, dtype=np.float32)
y_test_one = np.array(y_valid[:mid,:],   copy=True, dtype=np.float32)
x_test_two = np.array(x_valid[mid:,:,:], copy=True, dtype=np.float32)
y_test_two = np.array(y_valid[mid:,:],   copy=True, dtype=np.float32)
# %%
#! Cross-entropy problem if yo uwant to turn into classification.
# def custom_loss(y_true, y_pred):
#     """ Custom loss based on following formula:
#         sumN(0.5*(SoC-SoC*)^2)

#     Args:
#         y_true (tf.Tensor): True output values
#         y_pred (tf.Tensor): Predicted output from model

#     Returns:
#         tf.Tensor: The calculated Loss Value
#     """
#     y_pred = tf.convert_to_tensor(y_pred)
#     y_true = tf.cast(y_true, y_pred.dtype)
#     #print(f"True: {y_true[0]}" ) # \nvs Pred: {tf.make_ndarray(y_pred)}")
#     loss = tf.math.divide(
#                         x=tf.keras.backend.square(
#                                 x=tf.math.subtract(x=y_true,
#                                                    y=y_pred)
#                             ), 
#                         y=2
#                     )
#     #print("\nInitial Loss: {}".format(loss))    
#     #loss = 
#     #print(  "Summed  Loss: {}\n".format(loss))
#     return tf.keras.backend.sum(loss, axis=1)    

file_name : str = os.path.basename(__file__)[:-3]
model_loc : str = f'NewModels/{file_name}/{profile}-models/'
iEpoch = 0
firstLog : bool = True
iLr     : float = 0.001

# Disable AutoShard.
strategy = tf.distribute.MirroredStrategy()
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = \
                    tf.data.experimental.AutoShardPolicy.DATA
batch_size : int = 1
GLOBAL_BATCH_SIZE : int = batch_size*strategy.num_replicas_in_sync

ds_train = ds_train.batch(GLOBAL_BATCH_SIZE)
ds_valid = ds_valid.batch(GLOBAL_BATCH_SIZE)

ds_train = ds_train.with_options(options) #! Need document
ds_valid = ds_valid.with_options(options)

with strategy.scope():
    try:
        for _, _, files in os.walk(model_loc):
            for file in files:
                if file.endswith('.ch'):
                    iEpoch = int(os.path.splitext(file)[0])
        
        lstm_model : tf.keras.models.Sequential = tf.keras.models.load_model(
                f'{model_loc}{iEpoch}',
                compile=False)
        firstLog = False
        print("Model Identefied. Continue training.")
    except OSError as identifier:
        print("Model Not Found, creating new. {} \n".format(identifier))
        lstm_model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.InputLayer(input_shape=(x_train.shape[-2:]),
                                    batch_size=GLOBAL_BATCH_SIZE),
            tf.keras.layers.LSTM(
                units=500, activation='tanh', recurrent_activation='sigmoid',
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
        firstLog = True
    prev_model = tf.keras.models.clone_model(lstm_model,
                                    input_tensors=None, clone_function=None)

    checkpoints = tf.keras.callbacks.ModelCheckpoint(
        filepath =model_loc+f'{profile}-checkpoints/checkpoint',
        monitor='val_loss', verbose=0,
        save_best_only=False, save_weights_only=False,
        mode='auto', save_freq='epoch', options=None,
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=model_loc+
                f'tensorboard/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1, write_graph=True, write_images=False,
            update_freq='epoch', profile_batch=2, embeddings_freq=0,
            embeddings_metadata=None
        )

    nanTerminate = tf.keras.callbacks.TerminateOnNaN()

def tf_round(x : tf.Tensor, decimals : int = 0) -> tf.Tensor:
    """ Round to nearest decimal

    Args:
        x (tf.Tensor): Input value or array
        decimals (int, optional): How many precisions. Defaults to 0.

    Returns:
        tf.Tensor: Return rounded value
    """
    multiplier : tf.Tensor = tf.constant(
            value=10**decimals, dtype=x.dtype, shape=None,
            name='decimal_multiplier'
        )
    return tf.round(
                x=(x * multiplier), name='round_to_decimal'
            ) / multiplier

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
    if (iEpoch < 4):
        return iLr
    else:
        # lr = tf_round(x=lr * tf.math.exp(-0.2), decimals=5)
        lr = lr * tf.math.exp(-0.15)
        if lr >= 0.0002:
            return lr
        else:
            return  0.00005
    # return np.arange(0.001,0,-0.00002)[iEpoch]
    # return lr * 1 / (1 + decay * iEpoch)


# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='root_mean_squared_error', factor=0.2, patience=2, verbose=1,
#     mode='auto', min_delta=0.001, cooldown=0, min_lr=0.00005
# )
reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

# See how delta applies?
# loss: 1.5172e-05 -- 1.2151e-05
#  np.arange(0.001,0.00000,-0.00005)


with strategy.scope():
    lstm_model.compile(loss=tf.losses.MeanAbsoluteError(),
            optimizer=tf.optimizers.Adam(learning_rate=iLr,
                    beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False),
            metrics=[tf.metrics.MeanAbsoluteError(),
                        tf.metrics.RootMeanSquaredError(),
                        tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
            #run_eagerly=True
        )
    prev_model.compile(loss=tf.losses.MeanAbsoluteError(),
            optimizer=tf.optimizers.Adam(learning_rate=iLr,
                    beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False),
            metrics=[tf.metrics.MeanAbsoluteError(),
                        tf.metrics.RootMeanSquaredError(),
                        tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
        )
# %%
i_attempts : int = 0
n_attempts : int = 3
while iEpoch < mEpoch:
    iEpoch+=1
    print(f"Epoch {iEpoch}/{mEpoch}")
    
    # history = lstm_model.fit(x=x_train,y=y_train, epochs=1,
    #                     validation_data=(x_valid, y_valid),#ds_valid,
    #                     callbacks=[nanTerminate, reduce_lr],
    #                     batch_size=batch_size,
    #                     shuffle=True
    #                     )#! Initially Batch size 1; 8 is safe to run - 137s
    #!DS_TRAIN fails to update weights. Yet manages faster with iter()
    history = lstm_model.fit(x=ds_train, epochs=1,
                        validation_data=ds_valid,#ds_valid,
                        callbacks=[nanTerminate, reduce_lr],
                        batch_size=batch_size,
                        shuffle=True
                        )#! Initially Batch size 1; 8 is safe to run - 137s
    print(f'Learning Rate:{round(lstm_model.optimizer.lr.numpy(), 5)}')
    # history = lstm_model.fit(x=ds_train, epochs=1,
    #                     validation_data=ds_valid,
    #                     callbacks=[checkpoints], batch_size=1
    #                     )#! Initially Batch size 1; 8 is safe to run - 137s
    #? Dealing with NaN state. Give few trials to see if model improves
    if (tf.math.is_nan(history.history['loss'])):
        print('NaN model')
        while i_attempts < n_attempts:
            #! Hopw abut reducing input dataset. In this case, by half. Keeping
            #!only middle temperatures.
            print(f'Attempt {i_attempts}')
            #lstm_model.set_weights(prev_model.get_weights())
            lstm_model = tf.keras.models.clone_model(prev_model)
            lstm_model.compile(loss=tf.losses.MeanAbsoluteError(),
                    optimizer=tf.optimizers.Adam(learning_rate=iLr,
                            beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
                    metrics=[tf.metrics.MeanAbsoluteError(),
                            tf.metrics.RootMeanSquaredError(),
                            tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
                )
            history = lstm_model.fit(x=x_train[:,:,:], y=y_train[:,:], epochs=1,
                            validation_data=None,
                            callbacks=[nanTerminate],
                            batch_size=1, shuffle=True
                            )
            if (not tf.math.is_nan(history.history['loss'])):
                print(f'Attempt {i_attempts} Passed')
                break
            i_attempts += 1
        if (i_attempts == n_attempts) \
                and (tf.math.is_nan(history.history['loss'])):
            print("Model reaced the optimim -- Breaking")
            break
        else:
            lstm_model.save(filepath=f'{model_loc}{iEpoch}-{i_attempts}',
                            overwrite=True, include_optimizer=True,
                            save_format='h5', signatures=None, options=None,
                            save_traces=True
                )
            # lstm_model.save_weights(f'{model_loc}weights/{iEpoch}-{i_attempts}')
            i_attempts = 0
            #prev_model.set_weights(lstm_model.get_weights())
            prev_model = tf.keras.models.clone_model(lstm_model)
    else:
        #lstm_model.save(f'{model_loc}{iEpoch}')
        lstm_model.save(filepath=f'{model_loc}{iEpoch}',
                        overwrite=True, include_optimizer=True,
                        save_format='h5', signatures=None, options=None,
                        save_traces=True
                )
        # lstm_model.save_weights(f'{model_loc}weights/{iEpoch}')
        #prev_model.set_weights(lstm_model.get_weights())
        prev_model = tf.keras.models.clone_model(lstm_model)
    
    if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
        os.remove(f'{model_loc}{iEpoch-1}.ch')
    os.mknod(f'{model_loc}{iEpoch}.ch')
    
    # Saving history variable
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)
    # or save to csv:
    with open(f'{model_loc}history-{profile}.csv', mode='a') as f:
        if(firstLog):
            hist_df.to_csv(f, index=False)
            firstLog = False
        else:
            hist_df.to_csv(f, index=False, header=False)
    
    #! Run the Evaluate function
    #! Replace with tf.metric function.
    PRED = lstm_model.predict(x_valid, batch_size=1)
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_valid[::,]-PRED)))
    PERF = lstm_model.evaluate(x=x_valid,
                               y=y_valid,
                               batch_size=1,
                               verbose=0)
    # otherwise the right y-label is slightly clipped
    predicting_plot(profile=profile, file_name='Model №1',
                    model_loc=model_loc,
                    model_type='LSTM Train',
                    iEpoch=f'val-{iEpoch}',
                    Y=y_valid,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=PERF,
                    TAIL=y_valid.shape[0],
                    save_plot=True)
    if(PERF[-2] <=0.010): # Check thr RMSE
        print("RMS droped around 2.0%. Breaking the training")
        break
# %%
# TAIL=y_test_one.shape[0]
# PRED = lstm_model.predict(x_test_one, batch_size=1)
# RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
#             y_test_one[::skip,]-PRED)))
# vl_test_time = range(0,PRED.shape[0])
# fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
# ax1.plot(vl_test_time[:TAIL:skip], y_test_one[::skip,-1],
#         label="True", color='#0000ff')
# ax1.plot(vl_test_time[:TAIL:skip],
#         PRED,
#         label="Recursive prediction", color='#ff0000')

# ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
# ax1.set_xlabel("Time Slice (s)", fontsize=16)
# ax1.set_ylabel("SoC (%)", fontsize=16)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax2.plot(vl_test_time[:TAIL:skip],
#         RMS,
#         label="RMS error", color='#698856')
# ax2.fill_between(vl_test_time[:TAIL:skip],
#         RMS[:,0],
#             color='#698856')
# ax2.set_ylabel('Error', fontsize=16, color='#698856')
# ax2.tick_params(axis='y', labelcolor='#698856')
# if profile == 'DST':
#     ax1.set_title(f"{file_name} LSTM Test on US06 - {profile}-trained",
#                 fontsize=18)
# else:
#     ax1.set_title(f"{file_name} LSTM Test on DST - {profile}-trained",
#                 fontsize=18)
                
# ax1.legend(prop={'size': 16})
# ax1.set_ylim([-0.1,1.2])
# ax2.set_ylim([-0.1,1.6])
# fig.tight_layout()  # otherwise the right y-label is slightly clipped

# val_perf = lstm_model.evaluate(x=x_test_one,
#                                 y=y_test_one,
#                                 batch_size=1,
#                                 verbose=0)
# textstr = '\n'.join((
#     r'$Loss =%.2f$' % (val_perf[0], ),
#     r'$MAE =%.2f$' % (val_perf[1], ),
#     r'$RMSE=%.2f$' % (val_perf[2], )))
# ax1.text(0.85, 0.75, textstr, transform=ax1.transAxes, fontsize=18,
#         verticalalignment='top',
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
# fig.savefig(f'{model_loc}{profile}-test_One-{iEpoch}.svg')
# Cleaning Memory from plots
# fig.clf()
# plt.close()
PRED = lstm_model.predict(x_test_one, batch_size=1, verbose=1)
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(y_test_one[::,]-PRED)))
if profile == 'DST':
    predicting_plot(profile=profile, file_name='Model №1',
                    model_loc=model_loc,
                    model_type='LSTM Test on US06', iEpoch=f'Test One-{iEpoch}',
                    Y=y_test_one,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=lstm_model.evaluate(
                                    x=x_test_one,
                                    y=y_test_one,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_one.shape[0],
                    save_plot=True)
else:
    predicting_plot(profile=profile, file_name='Model №1',
                    model_loc=model_loc,
                    model_type='LSTM Test on DST', iEpoch=f'Test One-{iEpoch}',
                    Y=y_test_one,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=lstm_model.evaluate(
                                    x=x_test_one,
                                    y=y_test_one,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_one.shape[0],
                    save_plot=True)

# TAIL=y_test_two.shape[0]
# PRED = lstm_model.predict(x_test_two, batch_size=1)
# RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
#             y_test_two[::skip,]-PRED)))
# vl_test_time = range(0,PRED.shape[0])
# %%
def format_SoC(value, _):
    return int(value*100)
TAIL = y_test_one.shape[0]
test_time = np.linspace(0, PRED.shape[0]/60, PRED.shape[0])
skip=1
fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
ax1.plot(test_time[:TAIL:skip], y_test_one[::skip,-1],'-',
        label="Actual", color='#0000ff')
ax1.plot(test_time[:TAIL:skip],
        PRED,'--',
        label="Prediction", color='#ff0000')

# ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
ax1.set_xlabel("Time Slice (min)", fontsize=32)
ax1.set_ylabel("SoC (%)", fontsize=32)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(test_time[:TAIL:skip],
        RMS,
        label="ABS error", color='#698856')
ax2.fill_between(test_time[:TAIL:skip],
        RMS[:,0],
            color='#698856')
ax2.set_ylabel('Error', fontsize=32, color='#698856')
ax2.tick_params(axis='y', labelcolor='#698856')
# if profile == 'FUDS':
#     ax1.set_title(f"{file_name} LSTM Test on US06 - {profile}-trained",
#                 fontsize=18)
# else:
#     ax1.set_title(f"{file_name} LSTM Test on FUDS - {profile}-trained",
#                 fontsize=18)
ax1.legend(prop={'size': 32})
ax1.tick_params(axis='both', labelsize=28)
ax2.tick_params(axis='y', labelcolor='#698856', labelsize=28)
ax2.legend(loc='center right', bbox_to_anchor=(1.0,0.80), prop={'size': 32})
# ax1.set_ylim([-0.1,1.2])
# ax2.set_ylim([-0.1,1.6])
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
ax2.set_xlim([10,45])
ax2.set_ylim([-0.0,1.6])
ax1.set_title(
    f"Accuracy visualisation example",
    fontsize=36)
ax1.set_ylim([0.7,1])
ax1.set_xlim([10,50])
# ax1.annotate('Actual SoC percent', xy=(25, 0.86),
#             xycoords='data', fontsize=28, color='#0000ff',
#             xytext=(0.5, 0.85), textcoords='axes fraction',
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             horizontalalignment='right', verticalalignment='top',
#             )
# ax1.annotate('Predicted SoC', xy=(24, 0.77),
#             xycoords='data', fontsize=28, color='#ff0000',
#             xytext=(0.25, 0.25), textcoords='axes fraction',
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             horizontalalignment='right', verticalalignment='top',
#             )

plt.annotate(s='', xy=(25.5,0.83), xytext=(25.5,0.43),
                xycoords='data', fontsize=28, 
                arrowprops=dict(arrowstyle='<->', facecolor='black'),
                horizontalalignment='right', verticalalignment='top')
plt.annotate(s='', xy=(20.5,1.02), xytext=(20.5,0.68),
                xycoords='data', fontsize=28, 
                arrowprops=dict(arrowstyle='<->', facecolor='black'),
                horizontalalignment='right', verticalalignment='top')
ax1.text(19.5, 0.855, r'$\Delta$', fontsize=24)
ax1.text(24.5, 0.815, r'$\Delta$', fontsize=24)
ax1.annotate('Difference area fill', xy=(26, 0.82),
            xycoords='data', fontsize=28, color='#698856',
            xytext=(0.95, 0.45), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
ax1.annotate('', xy=(26, 0.72),
            xycoords='data', fontsize=28,
            xytext=(0.7, 0.41), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_SoC))
# val_perf = lstm_model.evaluate(x=x_test_two,
#                                 y=y_test_two,
#                                 batch_size=1,
#                                 verbose=0)
# textstr = '\n'.join((
#     r'$Loss =%.2f$' % (val_perf[0], ),
#     r'$MAE =%.2f$' % (val_perf[1], ),
#     r'$RMSE=%.2f$' % (val_perf[2], )))
# ax1.text(0.85, 0.75, textstr, transform=ax1.transAxes, fontsize=18,
#         verticalalignment='top',
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
fig.savefig(f'tests/figures/plot-example.svg')
# # Cleaning Memory from plots
# fig.clf()
# plt.close()
# %%
PRED = lstm_model.predict(x_test_two, batch_size=1, verbose=1)
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(y_test_two[::,]-PRED)))
if profile == 'FUDS':
    predicting_plot(profile=profile, file_name='Model №1',
                    model_loc=model_loc,
                    model_type='LSTM Test on US06', iEpoch=f'Test Two-{iEpoch}',
                    Y=y_test_two,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=lstm_model.evaluate(
                                    x=x_test_two,
                                    y=y_test_two,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_two.shape[0],
                    save_plot=True)
else:
    predicting_plot(profile=profile, file_name='Model №1',
                    model_loc=model_loc,
                    model_type='LSTM Test on FUDS', iEpoch=f'Test Two-{iEpoch}',
                    Y=y_test_two,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=lstm_model.evaluate(
                                    x=x_test_two,
                                    y=y_test_two,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_two.shape[0],
                    save_plot=True)
# %%
# Convert the model to Tensorflow Lite and save.
with open(f'{model_loc}Model-№1-{profile}.tflite', 'wb') as f:
    f.write(
        tf.lite.TFLiteConverter.from_keras_model(
                model=lstm_model
            ).convert()
        )