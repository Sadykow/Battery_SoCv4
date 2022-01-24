#!/usr/bin/python
# %% [markdown]
# # # 1
# # #
# # LSTM for SoC by GatethJavid 2020
# This version hjad no specification on technique used. Although, by the type
#and details in the second article published in Feb2021 - it is a stateless
#windowing technique.

#? This file used to train only on a FUDS dataset and validate against another
#?excluded FUDS datasets. In this case, out of 12 - 2 for validation.
#?Approximately 15~20% of provided data.
#? Testing performed on any datasets.
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
from py_modules.RobustAdam import RobustAdam
# from py_modules.SGOptimizer import SGOptimizer
from py_modules.utils import str2bool
from py_modules.plotting import predicting_plot
import time
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

opts = [('-d', 'False'), ('-e', '3'), ('-g', '0'), ('-p', 'FUDS')]
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
# dataGenerator = DataGenerator(train_dir=f'{Data}A123__Test',
#                               valid_dir=f'{Data}A123__Test',
#                               test_dir=f'{Data}A123__Test',
#                               columns=[
#                                 'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
#                                 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
#                                 ],
#                               PROFILE_range = profile)
# %%
window = WindowGenerator(Data=dataGenerator,
                        input_width=500, label_width=1, shift=0,
                        input_columns=['Current(A)', 'Voltage(V)',
                                                'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,
                        includeTarget=False, normaliseLabal=False,
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
# _, x_train, y_train = window.train
# %%
#return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)), axis=0))
#? DST - try 2 and 8
#? US06 - 7
#? FUDS - try 4 and 7
file_name : str = os.path.basename(__file__)[:-3]
model_loc : str = f'NewModels/{file_name}/{profile}-models/'
iEpoch    : int = 0
firstLog  : bool= True
try:
    for _, _, files in os.walk(model_loc):
        for file in files:
            if file.endswith('.ch'):
                iEpoch = int(os.path.splitext(file)[0])
    
    gru_model : tf.keras.models.Sequential = tf.keras.models.load_model(
            f'{model_loc}{iEpoch}',
            compile=False)
    firstLog = False
    print("Model Identefied. Continue training.")
except OSError as identifier:
    print("Model Not Found, creating new. {} \n".format(identifier))
    gru_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=x_train.shape[-2:]),
        tf.keras.layers.GRU(
            units=500, activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.2,
            recurrent_dropout=0.0, return_sequences=False, return_state=False,
            go_backwards=False, stateful=False, unroll=False, time_major=False,
            reset_after=True
        ),
        tf.keras.layers.Dense(units=1,
                              activation='sigmoid')
    ])
    firstLog = True
# %%
# MAE = tf.metrics.MeanAbsoluteError()
# RMSE = tf.metrics.RootMeanSquaredError()
# RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)

# optimiser = RobustAdam()
# loss_fn = tf.keras.losses.MeanAbsoluteError()
# pbar = tqdm(total=y_train.shape[0])
# # while iEpoch < mEpoch:
# #     iEpoch+=1
# for x, y in zip(np.expand_dims(x_train, axis=1), y_train):
#     with tf.GradientTape() as tape:
#         # Run the forward pass of the layer.
#         logits = gru_model(x, training=True)
#         # Compute the loss value 
#         loss_value = loss_fn(y_true=y, y_pred=logits)
        
#         grads = tape.gradient(loss_value, gru_model.trainable_weights)
#     # optimiser.update_labels()trainable_weights
#     optimiser.apply_gradients(zip(grads, gru_model.trainable_weights),
#                                 experimental_aggregate_gradients=True)
#     # Get matrics
#     MAE.update_state(y_true=y_train[:1], y_pred=logits)
#     RMSE.update_state(y_true=y_train[:1], y_pred=logits)
#     RSquare.update_state(y_true=y_train[:1], y_pred=logits)

#     # Progress Bar
#     pbar.update(1)
#     pbar.set_description(f' :: '
#                         f'loss: {loss_value:.4e} - '
#                         f'mae: {MAE.result():.4e} - '
#                         f'rmse: {RMSE.result():.4e} - '
#                         f'rsquare: {RSquare.result():04f}'
#                         )
# plt.plot(gru_model.predict(x_train, batch_size=1))
# plt.plot(y_train)
# %%
optimiser = RobustAdam(learning_rate = 0.001) #0.0001
loss_fn = tf.losses.MeanAbsoluteError()
MAE = tf.metrics.MeanAbsoluteError()
val_MAE = tf.metrics.MeanAbsoluteError()
RMSE = tf.metrics.RootMeanSquaredError()
val_RMSE = tf.metrics.RootMeanSquaredError()
RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)
val_RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)
loss_value : np.float32 = 1.0
# tf.optimizers.Adam()
# for w in gru_model.trainable_weights:
#     print(w)
#! We can potentialy run 2 models on single GPU getting to 86% utilisation.
#!Although, check if it safe. Use of tf.function speeds up training by 2.
# @tf.function
def train_single_st(x, y, prev_loss):
    with tf.GradientTape() as tape:
        logits = gru_model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, gru_model.trainable_weights)
    # 
    # print(f'Losses {prev_loss} and {loss_value}')
    optimiser.update_loss(prev_loss, loss_value)
    optimiser.apply_gradients(zip(grads, gru_model.trainable_weights))
    # Update learning rate
    optimiser.learning_rate = 0.0005
    # optimiser.get_config()
    MAE.update_state(y_true=y[:1], y_pred=logits)
    RMSE.update_state(y_true=y[:1], y_pred=logits)
    RSquare.update_state(y_true=y[:1], y_pred=logits)
    return loss_value

@tf.function
def test_step(x):
    return gru_model(x, training=False)

def valid_step(x, y):
    logits = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
    loss = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
    mae = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
    rmse = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
    rsquare = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
    # for i in trange(y.shape[0]):
    for i in range(y.shape[0]):
        logits[i] = test_step(x[i:i+1,:,:])
        val_MAE.update_state(y_true=y[i:i+1], y_pred=logits[i])
        val_RMSE.update_state(y_true=y[i:i+1], y_pred=logits[i])
        val_RSquare.update_state(y_true=y[i:i+1], y_pred=logits[i])
        loss[i] = loss_fn(y[i:i+1], logits[i])
        mae[i] = val_MAE.result()
        rmse[i] = val_RMSE.result()
        rsquare[i] = val_RSquare.result()
    #! Error with RMSE here. No mean should be used.
    return [np.mean(loss), np.mean(mae), np.mean(rmse), np.mean(rsquare), logits]
# # Disable AutoShard.
# strategy = tf.distribute.MirroredStrategy()
# # options = tf.data.Options()
# # options.experimental_distribute.auto_shard_policy = \
# #                     tf.data.experimental.AutoShardPolicy.DATA
# batch_size : int = strategy.num_replicas_in_sync
# worker_devices = strategy.extended.worker_devices

# def args_fn(context):
#     # print(ctx)
#     #! Return X and Y
#     x_train[i:i+1,:,:], y_train[i:i+1,:], loss_value
#     return tf.constant(1.0)

# distributed_values = strategy.experimental_distribute_values_from_function(
#                             args_fn
#                         )
# for i in range(2):      #! Lambda with train single st()
# #   loss_value = train_single_st(x_train[i:i+1,:,:], y_train[i:i+1,:], loss_value)
#   result = strategy.run(train_single_st, args=(distributed_values,))
#   print(result)
# %%
while iEpoch < mEpoch:
    iEpoch+=1
    pbar = tqdm(total=y_train.shape[0])
    tic : float = time.perf_counter()
    sh_i = np.arange(y_train.shape[0])
    np.random.shuffle(sh_i)
    print(f'Commincing Epoch: {iEpoch}')
    for i in sh_i[:]:
        loss_value = train_single_st(x_train[i:i+1,:,:], y_train[i:i+1,:], loss_value)
        # Progress Bar
        pbar.update(1)
        pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
                             f'loss: {loss_value:.4f} - '
                             f'mae: {MAE.result():.4f} - '
                             f'rmse: {RMSE.result():.4f} - '
                             f'rsquare: {RSquare.result():.4f}'
                            )
    pbar.close()
    toc : float = time.perf_counter() - tic
    print(f'Epoch {iEpoch}/{mEpoch} :: '
            f'Elapsed Time: {toc} - '
            f'loss: {loss_value:.4f} - '
            f'mae: {MAE.result():.4f} - '
            f'rmse: {RMSE.result():.4f} - '
            f'rsquare: {RSquare.result():.4f}'
        )
    # Saving model
    gru_model.save(filepath=f'{model_loc}{iEpoch}',
                overwrite=True, include_optimizer=True,
                save_format='h5', signatures=None, options=None,
                save_traces=True
        )
    if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
        os.remove(f'{model_loc}{iEpoch-1}.ch')
    os.mknod(f'{model_loc}{iEpoch}.ch')
    
    hist_df = pd.DataFrame(data={
            'loss'   : [np.array(loss_value)],
            'mae'    : [np.array(MAE.result())],
            'rmse'   : [np.array(RMSE.result())],
            'rsquare': [np.array(RSquare.result())]
        })
    
    PERF = valid_step(x_valid, y_valid)
    hist_df['vall_loss'] = PERF[0]
    hist_df['val_mean_absolute_error'] = PERF[1]
    hist_df['val_root_mean_squared_error'] = PERF[2]
    hist_df['val_r_square'] = PERF[3]
    
    hist_df['last_val_mean_absolute_error'] = np.array(val_MAE.result())
    hist_df['last_val_root_mean_squared_error'] = np.array(val_RMSE.result())
    hist_df['last_val_r_square'] = np.array(val_RSquare.result())
    # or save to csv:
    with open(f'{model_loc}history-{profile}.csv', mode='a') as f:
        if(firstLog):
            hist_df.to_csv(f, index=False)
            firstLog = False
        else:
            hist_df.to_csv(f, index=False, header=False)
    
    PRED = PERF[4]
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_valid[::,0]-PRED)))
    # otherwise the right y-label is slightly clipped
    predicting_plot(profile=profile, file_name='Model №5',
                    model_loc=model_loc,
                    model_type='GRU Train',
                    iEpoch=f'val-{iEpoch}',
                    Y=y_valid,
                    PRED=PRED,
                    RMS=np.expand_dims(RMS,axis=1),
                    val_perf=PERF[:4],
                    TAIL=y_valid.shape[0],
                    save_plot=True,
                    RMS_plot=False) #! Saving memory from high errors.
    if(PERF[-3] <=0.024): # Check thr RMSE
        print("RMS droped around 2.4%. Breaking the training")
        break
# %%
# tf.Tensor([0.00789516], shape=(1,), dtype=float32)
# tf.Tensor([1.910142e-06], shape=(1,), dtype=float32)
# %%
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# def loss(model, x, y, training):
#     # training=training is needed only if there are layers with different
#     # behavior during training versus inference (e.g. Dropout).
#     y_ = model(x, training=training)
#     return custom_loss(y_true=y, y_pred=y_)

# MAE = tf.metrics.MeanAbsoluteError()
# RMSE = tf.metrics.RootMeanSquaredError()
# RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)
# # RSquare = tfa.metrics.RSquare()
# optimiser = RobustAdam(lr_rate = 0.001,
#             beta_1 = 0.9, beta_2 = 0.999, beta_3 = 0.999, epsilon = 1e-7,
#             _is_first=False)
# while iEpoch < mEpoch:
#     iEpoch+=1
#     pbar = tqdm(total=y_train.shape[0])
#     # optimiser = RobustAdam(lr_rate = 0.001,
#     #             beta_1 = 0.9, beta_2 = 0.999, beta_3 = 0.999, epsilon = 1e-7,
#     #             _is_first=True)

#     # with tf.GradientTape() as tape:
#     #     # Run the forward pass of the layer.
#     #     logits = gru_model(x_train[:1,:,:], training=True)
#     #     # Compute the loss value 
#     #     loss_value = custom_loss(y_true=y_train[:1], y_pred=logits)
#     #     # 
#     #     grads = tape.gradient(loss_value, gru_model.trainable_variables)
#     # optimiser.apply_gradients(zip(grads, gru_model.trainable_variables),
#     #                             experimental_aggregate_gradients=True)
#     # # Get matrics
#     # MAE.update_state(y_true=y_train[:1], y_pred=logits)
#     # RMSE.update_state(y_true=y_train[:1], y_pred=logits)
#     # RSquare.update_state(y_true=y_train[:1], y_pred=logits)
#     # # Progress Bar
#     # pbar.update(1)
#     # pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
#     #                      f'loss: {loss_value[0]:.4e} - '
#     #                      f'mae: {MAE.result():.4e} - '
#     #                      f'rmse: {RMSE.result():.4e} - '
#     #                      f'rsquare: {RSquare.result():04f}'
#     #                     )

#     # optimiser = RobustAdam(lr_rate = 0.001,
#     #             beta_1 = 0.9, beta_2 = 0.999, beta_3 = 0.999, epsilon = 1e-7,
#     #             _is_first=False)
    
#     for x, y in zip(np.expand_dims(x_train[:,:,:], axis=1), y_train[:]):
#         with tf.GradientTape() as tape:
#             # Run the forward pass of the layer.
#             logits = gru_model(x, training=True)
#             # Compute the loss value 
#             loss_value = custom_loss(y_true=y, y_pred=logits)
#             # 
#             grads = tape.gradient(loss_value, gru_model.trainable_variables)
#         optimiser.apply_gradients(zip(grads, gru_model.trainable_variables),
#                                     experimental_aggregate_gradients=True)
#         # Get matrics
#         MAE.update_state(y_true=y_train[:1], y_pred=logits)
#         RMSE.update_state(y_true=y_train[:1], y_pred=logits)
#         RSquare.update_state(y_true=y_train[:1], y_pred=logits)

#         # Progress Bar
#         pbar.update(1)
#         pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
#                              f'loss: {loss_value[0]:.4e} - '
#                              f'mae: {MAE.result():.4e} - '
#                              f'rmse: {RMSE.result():.4e} - '
#                              f'rsquare: {RSquare.result():04f}'
#                             )  
        
#     # Evaluation
#     PRED = np.zeros(shape=(y_valid.shape[0],))
#     epoch_loss = np.zeros(shape=(y_valid.shape[0],))
#     epoch_mae = np.zeros(shape=(y_valid.shape[0],))
#     epoch_rmse = np.zeros(shape=(y_valid.shape[0],))
#     epoch_rsquare = np.zeros(shape=(y_valid.shape[0],))
    
#     for i in trange(0, len(y_valid)):
#         with tf.GradientTape() as tape:
#             y_pred = gru_model(x_valid[i:i+1,:,:], training=False)
#         MAE.update_state(y_valid[i], y_pred)
#         RMSE.update_state(y_valid[i], y_pred)
#         RSquare.update_state(y_valid[i:i+1], np.array(y_pred[0]))
#         PRED[i] = y_pred
#         epoch_loss[i] = custom_loss(y_valid[i], y_pred)
#         epoch_mae[i] = MAE.result()
#         epoch_rmse[i] = RMSE.result()
#         epoch_rsquare[i] = RSquare.result()
        
#     print(f'val_loss: {np.mean(epoch_loss)}')
#     print(f'val_mae: {np.mean(epoch_mae)}')
#     print(f'val_rmse: {np.mean(epoch_rmse)}')
#     print(f'val_rsquare: {np.mean(epoch_rsquare)}')
    
#     gru_model.save(filepath=f'{model_loc}{iEpoch}',
#                         overwrite=True, include_optimizer=True,
#                         save_format='h5', signatures=None, options=None,
#                         save_traces=True
#                 )
#     prev_model = tf.keras.models.clone_model(gru_model)

#     if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
#         os.remove(f'{model_loc}{iEpoch-1}.ch')
#     os.mknod(f'{model_loc}{iEpoch}.ch')
    
#     # Saving history variable
#     # convert the history.history dict to a pandas DataFrame:
#     hist_df = pd.DataFrame(data={
#             'loss' : [loss_value],
#             'val_loss' : [np.mean(epoch_loss)],
#             'val_mean_absolute_error' : [np.mean(epoch_mae)],
#             'val_root_mean_squared_error' : [np.mean(epoch_rmse)],
#             'val_r_square' : [np.mean(epoch_rsquare)],
#         })
#     # or save to csv:
#     with open(f'{model_loc}history-{profile}.csv', mode='a') as f:
#         if(firstLog):
#             hist_df.to_csv(f, index=False)
#             firstLog = False
#         else:
#             hist_df.to_csv(f, index=False, header=False)
    
#     RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
#                 y_valid[::,]-PRED)))
#     PERF = np.array([np.mean(epoch_loss), np.mean(epoch_mae),
#                      np.mean(epoch_rmse), np.mean(epoch_rsquare)],
#                      dtype=np.float32)

#     # otherwise the right y-label is slightly clipped
#     predicting_plot(profile=profile, file_name='Model №5',
#                     model_loc=model_loc,
#                     model_type='GRU Stateless - Train dataset',
#                     iEpoch=f'val-{iEpoch}',
#                     Y=y_valid,
#                     PRED=PRED,
#                     RMS=RMS,
#                     val_perf=PERF,
#                     TAIL=y_valid.shape[0],
#                     save_plot=True,
#                     RMS_plot=False)
#     if(PERF[-2] <=0.024): # Check thr RMSE
#         print("RMS droped around 2.4%. Breaking the training")
#         break
# %%
PRED = valid_step(x_test_one, y_test_one)
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(y_test_one[::,0]-PRED[4])))
if profile == 'DST':
    predicting_plot(profile=profile, file_name='Model №5',
                    model_loc=model_loc,
                    model_type='GRU Test on US06', iEpoch=f'Test One-{iEpoch}',
                    Y=y_test_one,
                    PRED=PRED[4],
                    RMS=np.expand_dims(RMS,axis=1),
                    val_perf=PRED[:4],
                    TAIL=y_test_one.shape[0],
                    save_plot=True)
else:
    predicting_plot(profile=profile, file_name='Model №5',
                    model_loc=model_loc,
                    model_type='GRU Test on DST', iEpoch=f'Test One-{iEpoch}',
                    Y=y_test_one,
                    PRED=PRED[4],
                    RMS=np.expand_dims(RMS,axis=1),
                    val_perf=PRED[:4],
                    TAIL=y_test_one.shape[0],
                    save_plot=True)
PRED = valid_step(x_test_two, y_test_two)
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(y_test_two[::,0]-PRED[4])))
if profile == 'FUDS':
    predicting_plot(profile=profile, file_name='Model №5',
                    model_loc=model_loc,
                    model_type='GRU Test on US06', iEpoch=f'Test Two-{iEpoch}',
                    Y=y_test_two,
                    PRED=PRED[4],
                    RMS=np.expand_dims(RMS,axis=1),
                    val_perf=PRED[:4],
                    TAIL=y_test_two.shape[0],
                    save_plot=True)
else:
    predicting_plot(profile=profile, file_name='Model №5',
                    model_loc=model_loc,
                    model_type='GRU Test on FUDS', iEpoch=f'Test Two-{iEpoch}',
                    Y=y_test_two,
                    PRED=PRED[4],
                    RMS=np.expand_dims(RMS,axis=1),
                    val_perf=PRED[:4],
                    TAIL=y_test_two.shape[0],
                    save_plot=True)
# %%
# Convert the model to Tensorflow Lite and save.
with open(f'{model_loc}Model-№5-{profile}.tflite', 'wb') as f:
    f.write(
        tf.lite.TFLiteConverter.from_keras_model(
                model=gru_model
            ).convert()
        )