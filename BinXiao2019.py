#!/usr/bin/python
# %% [markdown]
# # # 2
# # # 
# # GRU for SoC by Bin Xiao - 2019
# 

# Data windowing has been used as per: Ψ ={X,Y} where:
#Here,X_k=[I(k),V(k),T(k)] and Y_k=[SoC(k)], whereI(k),V(k),T(k) and SoC(k) are
#the current, voltage, temperature and SoC ofthe battery as measured at time
#step k.

# Compared with an LSTM-based RNN model,a GRU-based RNN model has a simpler 
#structure and fewer parameters, thus making model training easier. The GRU 
#structure is shown in Fig. 1

#* The architecture consis of Inut layer, GRU hidden, fullt conected Dense
#*and Output. Dropout applied at hidden layer.
#* Dense fully connected uses sigmoind activation.

#* Loss function standard MSE, I think. By the looks of it.

#* 2 optimizers for that:
#*  Nadam (Nesterov momentum into the Adam) b1=0.99
#*Remark 1:The purpose of the pre-training phase is to endow the GRU_RNN model
#*with the appropriate parametersto capture the inherent features of the 
#*training samples. The Nadam algorithm uses adaptive learning rates and
#*approximates the gradient by means of the Nesterov momentum,there by ensuring
#*fast convergence of the pre-training process.
#*  AdaMax (Extension to adam)
#*Remark 2:The purpose of the fine-tuning phase is to further adjust the
#*parameters to achieve greater accuracy bymeans of the AdaMax algorithm, which
#*converges to a morestable value.
#* Combine those two methods: Ensemle ptimizer.

#* Data scaling was performed using Min-Max equation:
#* x′=(x−min_x)/(max_x−min_x) - values between 0 and 1.

#* Test were applied separetly at 0,30,50C
#*RMSE,MAX,MAPE,R2
# %%
import os                       # OS, SYS, argc functions
import pandas as pd             # File read
import matplotlib as mpl        # Plot functionality
import matplotlib.pyplot as plt
import tensorflow as tf         # Tensorflow and Numpy replacement
import tensorflow_addons as tfa
import numpy as np
import logging

from sys import platform        # Get type of OS

from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
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
GPU=0
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
profile : str = 'd_DST'
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

# training = dataGenerator.train.loc[:, 
#                         ['Current(A)', 'Voltage(V)', 'Temperature (C)_1']]

# %%
window = WindowGenerator(Data=dataGenerator,
                        input_width=500, label_width=1, shift=0,
                        input_columns=['Current(A)', 'Voltage(V)', 'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,
                        includeTarget=False, normaliseLabal=False,
                        shuffleTraining=False)
_, xx_train, yy_train = window.train
_, xx_valid, yy_valid = window.valid
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

model_loc : str = f'Models/BinXiao2020/{profile}-models/'
iEpoch  : int = 0
p2 : int = 6
skipCompile1, skipCompile2 = False, False
try:
    for _, _, files in os.walk(model_loc):
        for file in files:
            if file.endswith('.ch'):
                iEpoch = int(os.path.splitext(file)[0])
    
    gru_model : tf.keras.models.Sequential = tf.keras.models.load_model(
            f'{model_loc}{iEpoch}',
            compile=True)
    print("Model Identefied. Continue training.")
except OSError as identifier:
    print("Model Not Found, creating new. {} \n".format(identifier))
    gru_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=x_train.shape[-2:],batch_size=None),
        tf.keras.layers.GRU(    #?260 by BinXia, times by 2 or 3
            units=560, activation='tanh', recurrent_activation='sigmoid',
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

# %%
mEpoch : int = 20
firtstEpoch : bool = True
#!DEBUG
#iEpoch = 9
while iEpoch < mEpoch:
    if (iEpoch<=p2 and not skipCompile1):
        gru_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001,
                    beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Nadam'
                    ),
                metrics=[tf.metrics.MeanAbsoluteError(),
                         tf.metrics.RootMeanSquaredError(),
                         tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
            )
        skipCompile1 = True
        print("\nOptimizer set: Nadam\n")
    elif (iEpoch>p2 and not skipCompile2):
        gru_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0005,
                    beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Adamax'
                    ),
                metrics=[tf.metrics.MeanAbsoluteError(),
                         tf.metrics.RootMeanSquaredError(),
                         tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
            )
        skipCompile2 = True
        print("\nOptimizer set: Adamax\n")
    iEpoch+=1
    print(f"Epoch {iEpoch}/{mEpoch}")
    
    history = gru_model.fit(x=x_train, y=y_train, epochs=1,
                        validation_data=(x_valid, y_valid),
                        callbacks=[checkpoints], batch_size=1, shuffle=True
                        )
    
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
            pd.DataFrame(data=hist_df.mean()).T.to_csv(f, index=False)
            firtstEpoch = False
        else:
            pd.DataFrame(data=hist_df.mean()).T.to_csv(f, index=False, header=False)
    
    #! Run the Evaluate function
    if(iEpoch % 10 == 0):
        skip=1
        TAIL=y_valid.shape[0]
        gru_model.reset_states()
        PRED = gru_model.predict(x_valid, batch_size=1)
        RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                    y_valid[::skip,]-PRED)))
        vl_test_time = range(0,PRED.shape[0])
        fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
        ax1.plot(vl_test_time[:TAIL:skip], y_valid[::skip],
                label="True", color='#0000ff')
        ax1.plot(vl_test_time[:TAIL:skip],
                PRED,
                label="Recursive prediction", color='#ff0000')

        ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
        ax1.set_xlabel("Time Slice (s)", fontsize=16)
        ax1.set_ylabel("SoC (%)", fontsize=16)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(vl_test_time[:TAIL:skip],
                RMS[:,0],
                label="RMS error", color='#698856')
        ax2.fill_between(vl_test_time[:TAIL:skip],
                RMS[:,0],
                    color='#698856')
        ax2.set_ylabel('Error', fontsize=16, color='#698856')
        ax2.tick_params(axis='y', labelcolor='#698856')
        ax1.set_title(f"BinXiao GRU Test 2019 - Valid dataset. {profile}-trained",
                    fontsize=18)
        ax1.legend(prop={'size': 16})
        ax1.set_ylim([-0.1,1.2])
        ax2.set_ylim([-0.1,1.6])
        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        val_perf = gru_model.evaluate(x=x_valid,
                                      y=y_valid,
                                      batch_size=1,
                                      verbose=0)
        textstr = '\n'.join((
            r'$Loss =%.2f$' % (val_perf[0], ),
            r'$MAE =%.2f$' % (val_perf[1], ),
            r'$RMSE=%.2f$' % (val_perf[2], )))
        ax1.text(0.85, 0.75, textstr, transform=ax1.transAxes, fontsize=18,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.savefig(f'{model_loc}{profile}-val-{iEpoch}.svg')
# %%
skip=1
start = 0
x_test = x_train[start:start+y_valid.shape[0],:,:]
y_test = y_train[start:start+y_valid.shape[0],:]
TAIL=y_test.shape[0]
PRED = gru_model.predict(x_test, batch_size=1)
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
            y_test[::skip,]-PRED)))
vl_test_time = range(0,PRED.shape[0])

fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
ax1.plot(vl_test_time[:TAIL:skip], y_test[::skip],
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
# ax2.fill_between(vl_test_time[:TAIL:skip],
#         RMS[:,0],
#             color='#698856')
ax2.set_ylabel('Error', fontsize=16, color='#698856')
ax2.tick_params(axis='y', labelcolor='#698856')
ax1.set_title(f"BinXiao GRU Test 2019 - Train dataset. {profile}-trained",
            fontsize=18)
ax1.legend(prop={'size': 16})
ax1.set_ylim([-0.1,1.2])
ax2.set_ylim([-0.1,1.6])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
gru_model.reset_states()
val_perf = gru_model.evaluate(x=x_test,
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
                model=gru_model
            ).convert()
        )