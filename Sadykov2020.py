#!/usr/bin/python
# %% [markdown]
# # Feed-forward implementation with 4 features.
# %%
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
from tqdm import tqdm, trange

from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
from py_modules.Attention import *
from py_modules.RobustAdam import RobustAdam
from cy_modules.utils import str2bool
from py_modules.plotting import predicting_plot
# %%
# Extract params
try:
    opts, args = getopt.getopt(sys.argv[1:],"hd:e:g:p:",
                    ["help", "debug=", "epochs=",
                     "gpu=", "profile="])
except getopt.error as err: 
    # output error, and return with an error code 
    print (str(err)) 
    print ('EXEPTION: Arguments requied!')
    sys.exit(2)

# opts = [('-d', 'False'), ('-e', '2'), ('-g', '1'), ('-p', 'DST')]
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
dataGenerator = DataGenerator(train_dir=f'{Data}A123_Matt_Set',
                              valid_dir=f'{Data}A123_Matt_Val',
                              test_dir=f'{Data}A123_Matt_Test',
                              columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                                'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                                ],
                              PROFILE_range = profile)
# %%
window = WindowGenerator(Data=dataGenerator,
                        input_width=500, label_width=1, shift=1,
                        input_columns=['Current(A)', 'Voltage(V)',
                                                'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,
                        includeTarget=True, normaliseLabal=False,
                        shuffleTraining=False)
ds_train, xx_train, yy_train = window.train
ds_valid, xx_valid, yy_valid = window.valid

# Entire Training set 
x_train = np.array(xx_train, copy=True, dtype=np.float32)
y_train = np.array(yy_train, copy=True, dtype=np.float32)

# For validation use same training
x_valid = np.array(xx_train[16800:25000,:,:], copy=True, dtype=np.float32)
y_valid = np.array(yy_train[16800:25000,:]  , copy=True, dtype=np.float32)

# For test dataset take the remaining profiles.
mid = int(xx_valid.shape[0]/2)+350
x_test_one = np.array(xx_valid[:mid,:,:], copy=True, dtype=np.float32)
y_test_one = np.array(yy_valid[:mid,:], copy=True, dtype=np.float32)
x_test_two = np.array(xx_valid[mid:,:,:], copy=True, dtype=np.float32)
y_test_two = np.array(yy_valid[mid:,:], copy=True, dtype=np.float32)
# %%
file_name : str = os.path.basename(__file__)[:-3]
model_loc : str = f'Models/{file_name}/{profile}-models/'
iEpoch = 0
firstLog  : bool = True
try:
    for _, _, files in os.walk(model_loc):
        for file in files:
            if file.endswith('.ch'):
                iEpoch = int(os.path.splitext(file)[0])
    
    lstm_model : tf.keras.models.Sequential = tf.keras.models.load_model(
            f'{model_loc}{iEpoch}',
            compile=False,
            custom_objects={"RSquare": tfa.metrics.RSquare,
                            "AttentionWithContext": AttentionWithContext,
                            "Addition": Addition,
                            })
    firstLog = False
    print("Model Identefied. Continue training.")
except OSError as identifier:
    print("Model Not Found, creating new. {} \n".format(identifier))
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.InputLayer(input_shape=x_train.shape[-2:]),
        tf.keras.layers.LSTM(
            units=510, activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            unit_forget_bias=True, kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.0, #!0.2
            recurrent_dropout=0.0, implementation=2, return_sequences=False, #!
            return_state=False, go_backwards=False, stateful=False,
            time_major=False, unroll=False
        ),
        # AttentionWithContext(),
        # Addition(),
        tf.keras.layers.Dense(units=1,
                            activation='sigmoid')
    ])
    firstLog = True
# prev_model = tf.keras.models.clone_model(lstm_model,
#                                     input_tensors=None, clone_function=None)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(
#         log_dir=model_loc+
#             f'tensorboard/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
#         histogram_freq=1, write_graph=True, write_images=False,
#         update_freq='epoch', profile_batch=2, embeddings_freq=0,
#         embeddings_metadata=None
#     )
# %%
# optimiser = RobustAdam(learning_rate = 0.0001)
optimiser = tf.optimizers.Adam(learning_rate = 0.001)
loss_fn   = tf.losses.MeanAbsoluteError()
@tf.function
def train_single_st(x, y, prev_loss):
    with tf.GradientTape() as tape:
        logits     = lstm_model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, lstm_model.trainable_weights)
    # optimiser.update_loss(prev_loss, loss_value)
    optimiser.apply_gradients(zip(grads, lstm_model.trainable_weights))
    MAE.update_state(y_true=y[:1], y_pred=logits)
    RMSE.update_state(y_true=y[:1], y_pred=logits)
    RSquare.update_state(y_true=y[:1], y_pred=logits)
    return loss_value

@tf.function
def test_step(x):
    return lstm_model(x, training=False)

def valid_step(x, y):
    logits  = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
    loss    = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
    mae     = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
    rmse    = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
    rsquare = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
    for i in trange(y.shape[0]):
        logits[i] = test_step(x[i:i+1,:,:])
        MAE.update_state(y_true=y[i:i+1], y_pred=logits[i])
        RMSE.update_state(y_true=y[i:i+1], y_pred=logits[i])
        RSquare.update_state(y_true=y[i:i+1], y_pred=logits[i])
        loss[i]    = loss_fn(y[i:i+1], logits[i])
        mae[i]     = MAE.result()
        rmse[i]    = RMSE.result()
        rsquare[i] = RSquare.result()
    return [np.mean(loss), np.mean(mae), np.mean(rmse), np.mean(rsquare), logits]

MAE     = tf.metrics.MeanAbsoluteError()
RMSE    = tf.metrics.RootMeanSquaredError()
RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)
loss_value : np.float32 = 1.0
while iEpoch < mEpoch:
    iEpoch+=1
    pbar = tqdm(total=y_train.shape[0])
    sh_i = np.arange(y_train.shape[0])
    np.random.shuffle(sh_i)
    for i in sh_i[:]:
        loss_value = train_single_st(x_train[i:i+1,:,:], y_train[i:i+1,:],
                                        loss_value)
        # Progress Bar
        pbar.update(1)
        pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
                            f'loss: {loss_value:.4f} - '
                            f'mae: {MAE.result():.4f} - '
                            f'rmse: {RMSE.result():.4f} - '
                            f'rsquare: {RSquare.result():.4f}'
                            )
    pbar.close()
    # Saving model
    lstm_model.save(filepath=f'{model_loc}{iEpoch}',
                overwrite=True, include_optimizer=True,
                save_format='h5', signatures=None, options=None,
                save_traces=True
        )
    if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
        os.remove(f'{model_loc}{iEpoch-1}.ch')
    os.mknod(f'{model_loc}{iEpoch}.ch')
    
    PERF = valid_step(x_valid, y_valid)
    hist_df = pd.DataFrame(data={
            'loss'   : [np.array(loss_value)],
            'mae'    : [np.array(MAE.result())],
            'rmse'   : [np.array(RMSE.result())],
            'rsquare': [np.array(RSquare.result())]
        })
    hist_df['vall_loss'] = PERF[0]
    hist_df['val_mean_absolute_error'] = PERF[1]
    hist_df['val_root_mean_squared_error'] = PERF[2]
    hist_df['val_r_square'] = PERF[3]
    
    # or save to csv:
    with open(f'{model_loc}history-{profile}.csv', mode='a') as f:
        if(firstLog):
            hist_df.to_csv(f, index=False)
            firstLog = False
        else:
            hist_df.to_csv(f, index=False, header=False)
    
    PRED = PERF[4]
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_valid[::,]-PRED)))
    # otherwise the right y-label is slightly clipped
    predicting_plot(profile=profile, file_name='myModel №1',
                    model_loc=model_loc,
                    model_type='LST Test - Train dataset',
                    iEpoch=f'val-{iEpoch}',
                    Y=y_valid,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=PERF[:4],
                    TAIL=y_valid.shape[0],
                    save_plot=True,
                    RMS_plot=False) #! Saving memory from high errors.
    if(PERF[-3] <=0.012): # Check thr RMSE
        print("RMS droped around 1.2%. Breaking the training")
        break
# %%
# VIT_input = x_valid[0,:,:3]
# SOC_input = x_valid[0,:,3:]
# PRED = np.zeros(shape=(y_valid.shape[0],), dtype=np.float32)
# for i in trange(y_valid.shape[0]):
#     logits = lstm_model.predict(
#                             x=np.expand_dims(
#                                 np.concatenate(
#                                     (VIT_input, SOC_input),
#                                     axis=1),
#                                 axis=0),
#                             batch_size=1
#                         )
#     VIT_input = x_valid[i,:,:3]
#     SOC_input = np.concatenate(
#                         (SOC_input, logits),
#                         axis=0)[1:,:]
#     PRED[i] = logits
# plt.plot(y_valid[:,0])
# plt.plot(PRED)
# %%
# log_dir=model_loc+ \
#     f'tensorboard/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

# tf.profiler.experimental.start(log_dir)
# for step in trange(10):
#     with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
#         # logit = test_step(x_train[step:step+1,:,:])
#         # loss_value = loss_fn(y_train[step:step+1], logit)
#         loss_value = train_single_st(x_train[step:step+1,:,:],
#                                         y_train[step:step+1,:],
#                                         loss_value)
#     pbar.set_description(f'TensorBoard Train :: '
#                             f'loss: {loss_value:.4f} - '
#                             f'mae: {MAE.result():.4f} - '
#                             f'rmse: {RMSE.result():.4f} - '
#                             f'rsquare: {RSquare.result():.4f}'
#                         )
# tf.profiler.experimental.stop()
