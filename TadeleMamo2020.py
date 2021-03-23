#!/usr/bin/python
# %% [markdown]
# # # 2
# # #
# # LSTM with Attention Mechanism for SoC by Tadele Mamo - 2020
# 

# Data windowing has been used as per: Ψ ={X,Y} where:
# %%
import os                       # OS, SYS, argc functions
import pandas as pd             # File read
import matplotlib as mpl        # Plot functionality
import matplotlib.pyplot as plt
import tensorflow as tf

import tensorflow_addons as tfa
import tensorflow_probability as tfp
import numpy as np
import logging

from sys import platform        # Get type of OS

from extractor.WindowGenerator import WindowGenerator
from extractor.DataGenerator import *
from modules.Attention import *

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
GPU=1
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
profile : str = 'US06'
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
                        input_columns=['Current(A)', 'Voltage(V)', 'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,
                        includeTarget=False, normaliseLabal=True,
                        shuffleTraining=False)
ds_train, xx_train, yy_train = window.train
ds_valid, xx_valid, yy_valid = window.valid
x_train = np.array(xx_train, copy=True, dtype=np.float32)
x_valid = np.array(xx_valid, copy=True, dtype=np.float32)
y_train = np.array(yy_train, copy=True, dtype=np.float32)
y_valid = np.array(yy_valid, copy=True, dtype=np.float32)

# %%
def custom_loss(y_true, y_pred):
    #! No custom loss used in this implementation
    #!Used standard mean_squared_error()
    y_pred = tf.framework.ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = tf.framework.ops.math_ops.cast(y_true, y_pred.dtype)
    return tf.keras.backend.mean(tf.ops.math_ops.squared_difference(y_pred, y_true), axis=-1)

model_loc : str = f'Models/TadeleMamo2020/{profile}-models/'
iEpoch = 0
try:
    for _, _, files in os.walk(model_loc):
        for file in files:
            if file.endswith('.ch'):
                iEpoch = int(os.path.splitext(file)[0])
    
    lstm_model : tf.keras.models.Sequential = tf.keras.models.load_model(
            f'{model_loc}{iEpoch}',
            compile=False)
    print("Model Identefied. Continue training.")
except OSError as identifier:
    print("Model Not Found, creating new. {} \n".format(identifier))
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.InputLayer(input_shape=x_train.shape[-2:]),
        tf.keras.layers.LSTM(
            units=500, activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            unit_forget_bias=True, kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.2,
            recurrent_dropout=0.0, implementation=2, return_sequences=True, #!
            return_state=False, go_backwards=False, stateful=False,
            time_major=False, unroll=False#,batch_input_shape=(1, 500, 3)
        ),
        AttentionWithContext(),
        Addition(),
        #tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1,
                            activation=None)
    ])

checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath =model_loc+f'{profile}-checkpoints/checkpoint',
    monitor='val_loss', verbose=0,
    save_best_only=False, save_weights_only=False,
    mode='auto', save_freq='epoch', options=None,
)
lstm_model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(),
            optimizer=tf.optimizers.Adam(learning_rate=0.001,
                    beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
            metrics=[tf.metrics.MeanAbsoluteError(),
                     tf.metrics.RootMeanSquaredError(),
                     tf.metrics.MeanAbsolutePercentageError(),
                     tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
            #run_eagerly=True
            )
# lstm_model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(),
#             # optimizer=tfp.optimizer.differential_evolution_minimize(
#             #         tf.keras.losses.MeanAbsolutePercentageError(),
#             #         initial_population=tf.constant([5,10,20]),
#             #         initial_position=None, #tf.constant([30,100,140]),
#             #         population_size=50,
#             #         crossover_prob=0.9,

#             #         population_stddev=0.8,#1.0,
                    
#             #         max_iterations=100, func_tolerance=0,
#             #         position_tolerance=1,
#             #         differential_weight=tf.Variable(1),
#             #         seed=None, name=None
#             #     ),
#             optimizer=tfp.optimizer.differential_evolution_one_step(
#                 tf.keras.losses.MeanAbsolutePercentageError(),
#                 tf.constant([5,10,20]),
#                 population_values=[30,100,140],
#                 differential_weight=0.5,
#                 crossover_prob=0.9, seed=None, name=None
#             ),
#             metrics=[tf.metrics.MeanAbsoluteError(),
#                      tf.metrics.RootMeanSquaredError(),
#                      tf.metrics.MeanAbsolutePercentageError(),
#                      tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
#             #run_eagerly=True
#             )
# %%
mEpoch : int = 40
firtstEpoch : bool = True
while iEpoch < mEpoch:
    iEpoch+=1
    print(f"Epoch {iEpoch}/{mEpoch}")
#! In Mamo methods implement Callback to reset model after 500 steps and then 
#!step by one sample for next epoch to capture shift in data. Hell method, but
#!might be more effective that batching 12 together.  
    history = lstm_model.fit(x=x_train, y=y_train, epochs=1,
                        validation_data=(x_valid, y_valid),
                        callbacks=[checkpoints], batch_size=1, shuffle=True
                        )#! Initially Batch size 1; 8 is safe to run - 137s
    lstm_model.save(f'{model_loc}{iEpoch}')
    lstm_model.save_weights(f'{model_loc}weights/{iEpoch}')
    
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
    
    #! Run the Evaluate function
    if(iEpoch % 10 == 0):
        skip=1
        TAIL=y_valid.shape[0]
        PRED = lstm_model.predict(x_valid)
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
        ax1.set_title("Chenali LSTM Test 2017 - Valid dataset. FUDS-trained",
                    fontsize=18)
        ax1.legend(prop={'size': 16})
        ax1.set_ylim([-0.1,1.2])
        ax2.set_ylim([-0.1,1.6])
        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        val_perf = lstm_model.evaluate(x=x_valid,
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
skip=1
start = 0
x_test = x_train[start:start+y_valid.shape[0],:,:]
y_test = y_train[start:start+y_valid.shape[0],:]
TAIL=y_test.shape[0]
PRED = lstm_model.predict(x_test, batch_size=1)
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
            y_test[::skip,]-PRED)))
vl_test_time = range(0,PRED.shape[0])
fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
ax1.plot(vl_test_time[:TAIL:skip], y_test[::skip,-1],
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
ax1.set_title(f"Chenali LSTM Test 2017 - Train dataset. {profile}-trained",
            fontsize=18)
ax1.legend(prop={'size': 16})
ax1.set_ylim([-0.1,1.2])
ax2.set_ylim([-0.1,1.6])
fig.tight_layout()  # otherwise the right y-label is slightly clipped

val_perf = lstm_model.evaluate(x=x_test,
                                y=y_test,
                                batch_size=1,
                                verbose=0)
textstr = '\n'.join((
    r'$Loss =%.2f$' % (val_perf[0], ),
    r'$MAE =%.2f$' % (val_perf[1], ),
    r'$RMSE=%.2f$' % (val_perf[2], )))
ax1.text(0.85, 0.75, textstr, transform=ax1.transAxes, fontsize=18,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
fig.savefig(f'{model_loc}{profile}-train-{iEpoch}.svg')

# %%
# Convert the model to Tensorflow Lite and save.
with open(f'{model_loc}{profile}.tflite', 'wb') as f:
    f.write(
        tf.lite.TFLiteConverter.from_keras_model(
                model=lstm_model
            ).convert()
        )