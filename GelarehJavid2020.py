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

from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
from modules.RobustAdam import RobustAdam
from extractor.utils import str2bool
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

# opts = [('-d', 'False'), ('-e', '50'), ('-g', '1'), ('-p', 'DST')]
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
# %%
def custom_loss(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)), axis=0))

file_name : str = os.path.basename(__file__)[:-3]
model_loc : str = f'Models/{file_name}/{profile}-models/'
iEpoch = 0
try:
    for _, _, files in os.walk(model_loc):
        for file in files:
            if file.endswith('.ch'):
                iEpoch = int(os.path.splitext(file)[0])
    
    gru_model : tf.keras.models.Sequential = tf.keras.models.load_model(
            f'{model_loc}{iEpoch}',
            compile=False)
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
# gru_model.compile(loss=custom_loss,
#             optimizer=RobustAdam(learning_rate = 0.001,
#                  beta_1 = 0.9, beta_2 = 0.999, beta_3 = 0.999, epsilon = 1e-7,
#                  cost = custom_loss),
#             metrics=[tf.metrics.MeanAbsoluteError(),
#                      tf.metrics.RootMeanSquaredError(),
#                      tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float64)],
#             #run_eagerly=True
#             )
# %%
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    return custom_loss(y_true=y, y_pred=y_)

MAE = tf.metrics.MeanAbsoluteError()
RMSE = tf.metrics.RootMeanSquaredError()
RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)

firtstEpoch : bool = True
while iEpoch < mEpoch:
    iEpoch+=1
    optimiser = RobustAdam(lr_rate = 0.001,
                beta_1 = 0.9, beta_2 = 0.999, beta_3 = 0.999, epsilon = 1e-7,
                _is_first=True)

    with tf.GradientTape() as tape:
        loss_value = loss(gru_model, x_train[:1,:,:], y_train[:1], training=True)
        grads = tape.gradient(loss_value, gru_model.trainable_variables)
    optimiser.apply_gradients(zip(grads, gru_model.trainable_variables),
                                experimental_aggregate_gradients=True)

    size = int(y_train.shape[0]/2)
    optimiser = RobustAdam(lr_rate = 0.001,
                beta_1 = 0.9, beta_2 = 0.999, beta_3 = 0.999, epsilon = 1e-7,
                _is_first=False)
    for x, y in zip(np.expand_dims(x_train[1:size,:,:], axis=1), y_train[1:size]):
        with tf.GradientTape() as tape:
            #current_loss = custom_loss(gru_model(x_train[:1,:,:]))
            loss_value = loss(gru_model, x, y, training=True)
            grads = tape.gradient(loss_value, gru_model.trainable_variables)
        #print(f'LossValue: {loss_value}, True {y[0]}')
        optimiser.apply_gradients(zip(grads, gru_model.trainable_variables),
                                    experimental_aggregate_gradients=True)

    print(f'LossValue: {loss_value}, True {y[0]}')
    
    # Evaluation
    losses = np.zeros(shape=(y_valid.shape[0],))
    epoch_mae = np.zeros(shape=(y_valid.shape[0],))
    epoch_rmse = np.zeros(shape=(y_valid.shape[0],))
    epoch_rsquare = np.zeros(shape=(y_valid.shape[0],))
    
    size = int(y_valid.shape[0]/2)
    i = 0
    for x, y in zip(np.expand_dims(x_valid[:size,:,:], axis=1), y_valid[:size]):
        with tf.GradientTape() as tape:
            losses[i] = loss(gru_model, x, y, training=False)
        MAE.update_state(losses[i])
        RMSE.update_state(losses[i])
        RSquare.update_state(losses[i])
        epoch_mae[i] = MAE.result()
        epoch_rmse[i] = RMSE.result()
        epoch_rsquare[i] = RSquare.result()
        i += 1

    print(f'val_loss: {np.mean(losses)}')
    print(f'val_mae: {np.mean(epoch_mae)}')
    print(f'val_rmse: {np.mean(epoch_rmse)}')
    print(f'val_rsquare: {np.mean(epoch_rsquare)}')
    
    gru_model.save(f'{model_loc}{iEpoch}')
    gru_model.save_weights(f'{model_loc}weights/{iEpoch}')
    
    if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
        os.remove(f'{model_loc}{iEpoch-1}.ch')
    os.mknod(f'{model_loc}{iEpoch}.ch')
    
    # Saving history variable
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)
    # or save to csv:
    with open(f'{model_loc}history-{profile}.csv', mode='a') as f:
        if(firtstEpoch):
            hist_df.to_csv(f, index=False)
            firtstEpoch = False
        else:
            hist_df.to_csv(f, index=False, header=False)
    
    skip=1
    TAIL=y_valid.shape[0]
    PRED = gru_model.predict(x_valid,batch_size=1)
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_valid[::skip,]-PRED)))
    vl_test_time = range(0,PRED.shape[0])
    fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
    ax1.plot(vl_test_time[:TAIL:skip], y_valid[::skip,],
            label="True", color='#0000ff')
    ax1.plot(vl_test_time[:TAIL:skip],
            PRED,
            label="Recursive prediction", color='#ff0000')

    ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
    ax1.set_xlabel("Time Slice (s)", fontsize=16)
    ax1.set_ylabel("SoC (%)", fontsize=16)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(vl_test_time[:TAIL:skip],
            RMS,
            label="RMS error", color='#698856')
    ax2.fill_between(vl_test_time[:TAIL:skip],
            RMS[:,0],
                color='#698856')
    ax2.set_ylabel('Error', fontsize=16, color='#698856')
    ax2.tick_params(axis='y', labelcolor='#698856')
    ax1.set_title(f"{file_name} LSTM Test - Train dataset. {profile}-{iEpoch}",
                fontsize=18)
    ax1.legend(prop={'size': 16})
    ax1.set_ylim([-0.1,1.2])
    ax2.set_ylim([-0.1,1.6])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    val_perf = gru_model.evaluate(x=x_valid,
                                y=y_valid,
                                verbose=0)
    textstr = '\n'.join((
        r'$Loss =%.2f$' % (val_perf[0], ),
        r'$MAE =%.2f$' % (val_perf[1], ),
        r'$RMSE=%.2f$' % (val_perf[2], )))
    ax1.text(0.85, 0.75, textstr, transform=ax1.transAxes, fontsize=18,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.savefig(f'{model_loc}{profile}val-{iEpoch}.svg')
# %%
TAIL=y_test_one.shape[0]
PRED = gru_model.predict(x_test_one, batch_size=1)
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
            y_test_one[::skip,]-PRED)))
vl_test_time = range(0,PRED.shape[0])
fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
ax1.plot(vl_test_time[:TAIL:skip], y_test_one[::skip,-1],
        label="True", color='#0000ff')
ax1.plot(vl_test_time[:TAIL:skip],
        PRED,
        label="Recursive prediction", color='#ff0000')

ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
ax1.set_xlabel("Time Slice (s)", fontsize=16)
ax1.set_ylabel("SoC (%)", fontsize=16)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(vl_test_time[:TAIL:skip],
        RMS,
        label="RMS error", color='#698856')
ax2.fill_between(vl_test_time[:TAIL:skip],
        RMS[:,0],
            color='#698856')
ax2.set_ylabel('Error', fontsize=16, color='#698856')
ax2.tick_params(axis='y', labelcolor='#698856')
if profile == 'DST':
    ax1.set_title(f"{file_name} LSTM Test on US06 - {profile}-trained",
                fontsize=18)
else:
    ax1.set_title(f"{file_name} LSTM Test on DST - {profile}-trained",
                fontsize=18)
                
ax1.legend(prop={'size': 16})
ax1.set_ylim([-0.1,1.2])
ax2.set_ylim([-0.1,1.6])
fig.tight_layout()  # otherwise the right y-label is slightly clipped

val_perf = gru_model.evaluate(x=x_test_one,
                                y=y_test_one,
                                batch_size=1,
                                verbose=0)
textstr = '\n'.join((
    r'$Loss =%.2f$' % (val_perf[0], ),
    r'$MAE =%.2f$' % (val_perf[1], ),
    r'$RMSE=%.2f$' % (val_perf[2], )))
ax1.text(0.85, 0.75, textstr, transform=ax1.transAxes, fontsize=18,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
fig.savefig(f'{model_loc}{profile}-test_One-{iEpoch}.svg')
# %%
TAIL=y_test_two.shape[0]
PRED = gru_model.predict(x_test_two, batch_size=1)
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
            y_test_two[::skip,]-PRED)))
vl_test_time = range(0,PRED.shape[0])
fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
ax1.plot(vl_test_time[:TAIL:skip], y_test_two[::skip,-1],
        label="True", color='#0000ff')
ax1.plot(vl_test_time[:TAIL:skip],
        PRED,
        label="Recursive prediction", color='#ff0000')

ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
ax1.set_xlabel("Time Slice (s)", fontsize=16)
ax1.set_ylabel("SoC (%)", fontsize=16)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(vl_test_time[:TAIL:skip],
        RMS,
        label="RMS error", color='#698856')
ax2.fill_between(vl_test_time[:TAIL:skip],
        RMS[:,0],
            color='#698856')
ax2.set_ylabel('Error', fontsize=16, color='#698856')
ax2.tick_params(axis='y', labelcolor='#698856')
if profile == 'FUDS':
    ax1.set_title(f"{file_name} LSTM Test on US06 - {profile}-trained",
                fontsize=18)
else:
    ax1.set_title(f"{file_name} LSTM Test on FUDS - {profile}-trained",
                fontsize=18)

ax1.legend(prop={'size': 16})
ax1.set_ylim([-0.1,1.2])
ax2.set_ylim([-0.1,1.6])
fig.tight_layout()  # otherwise the right y-label is slightly clipped

val_perf = gru_model.evaluate(x=x_test_two,
                                y=y_test_two,
                                batch_size=1,
                                verbose=0)
textstr = '\n'.join((
    r'$Loss =%.2f$' % (val_perf[0], ),
    r'$MAE =%.2f$' % (val_perf[1], ),
    r'$RMSE=%.2f$' % (val_perf[2], )))
ax1.text(0.85, 0.75, textstr, transform=ax1.transAxes, fontsize=18,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
fig.savefig(f'{model_loc}{profile}-test_Two-{iEpoch}.svg')
# %%
# Convert the model to Tensorflow Lite and save.
with open(f'{model_loc}{profile}.tflite', 'wb') as f:
    f.write(
        tf.lite.TFLiteConverter.from_keras_model(
                model=gru_model
            ).convert()
        )