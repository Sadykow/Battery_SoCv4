#!/usr/bin/python
# %% [markdown]
# # # 1
# # #
# # LSTM for SoH by Wei Zhang
# %%
import datetime
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

from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
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

# opts = [('-d', 'False'), ('-e', '5'), ('-g', '0'), ('-p', 'DST')]
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
y_test_one = np.array(yy_valid[:mid,:],   copy=True, dtype=np.float32)
x_test_two = np.array(xx_valid[mid:,:,:], copy=True, dtype=np.float32)
y_test_two = np.array(yy_valid[mid:,:],   copy=True, dtype=np.float32)
# %%
#? Root Mean Squared Error loss function
custom_loss = lambda y_true, y_pred: tf.sqrt(
            x=tf.reduce_mean(
                    input_tensor=tf.square(
                            x=tf.subtract(
                                x=tf.cast(x=y_true, dtype=y_pred.dtype),
                                y=tf.convert_to_tensor(value=y_pred)

                            )
                        ),
                    axis=0,
                    keepdims=False
                )
        )
file_name : str = os.path.basename(__file__)[:-3]
model_loc : str = f'Models/{file_name}/{profile}-models/'
iEpoch = 0
firstLog : bool = True
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
        tf.keras.layers.InputLayer(input_shape=x_train.shape[-2:],
                                   batch_size=None),
        tf.keras.layers.LSTM(
            units=200, activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            unit_forget_bias=True, kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.3,
            recurrent_dropout=0.0, implementation=2, return_sequences=True, #!
            return_state=False, go_backwards=False, stateful=False,
            time_major=False, unroll=False#,batch_input_shape=(1, 500, 3)
        ),
        tf.keras.layers.LSTM(
            units=200, activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            unit_forget_bias=True, kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.2,
            recurrent_dropout=0.0, implementation=2, return_sequences=False, #!
            return_state=False, go_backwards=False, stateful=False,
            time_major=False, unroll=False#,batch_input_shape=(1, 500, 3)
        ),
        tf.keras.layers.Dense(units=100, activation='relu', use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            bias_constraint=None),
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


# lstm_model.compile(loss=custom_loss,
#         optimizer=tf.optimizers.Adam(learning_rate=0.001,
#                 beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
#         metrics=[tf.metrics.MeanAbsoluteError(),
#                     tf.metrics.RootMeanSquaredError(),
#                     tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
#         #run_eagerly=True
#     )
# prev_model.compile(loss=custom_loss,
#         optimizer=tf.optimizers.Adam(learning_rate=0.001,
#                 beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
#         metrics=[tf.metrics.MeanAbsoluteError(),
#                     tf.metrics.RootMeanSquaredError(),
#                     tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
#     )
# %%
MAE = tf.metrics.MeanAbsoluteError()
RMSE = tf.metrics.RootMeanSquaredError()
RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)

optimiser=tf.optimizers.Adam(learning_rate=0.0001,
                beta_1=0.9, beta_2=0.999, epsilon=10e-08)

while iEpoch < mEpoch:
    iEpoch+=1
    i : int = 0
    epoch_loss = np.zeros(shape=(y_train.shape[0],))
    epoch_mae = np.zeros(shape=(y_train.shape[0],))
    epoch_rmse = np.zeros(shape=(y_train.shape[0],))
    epoch_rsquare = np.zeros(shape=(y_train.shape[0],))
    pbar = tqdm(total=y_train.shape[0])
    for x, y in zip(np.expand_dims(x_train[:,:,:], axis=1), y_train[:]):
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            logits = lstm_model(x, training=True)
            # Compute the loss value 
            loss_value = custom_loss(y_true=y, y_pred=logits)
            # 
            grads = tape.gradient(loss_value, lstm_model.trainable_variables)
        optimiser.apply_gradients(zip(grads, lstm_model.trainable_variables),
                                    experimental_aggregate_gradients=True)
        # Get matrics
        MAE.update_state(y_true=y_train[:1], y_pred=logits)
        RMSE.update_state(y_true=y_train[:1], y_pred=logits)
        RSquare.update_state(y_true=y_train[:1], y_pred=logits)
        epoch_loss[i] = loss_value
        epoch_mae[i] = MAE.result()
        epoch_rmse[i] = RMSE.result()
        epoch_rsquare[i] = RSquare.result()
        
        # Progress Bar
        pbar.update(1)
        pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
                             f'loss: {epoch_loss[i]:.4e} - '
                             f'mae: {epoch_mae[i]:.4e} - '
                             f'rmse: {epoch_rmse[i]:.4e} - '
                             f'rsquare: {epoch_rsquare[i]:04f}'
                            )
        i+=1

    # Evaluation
    PRED = np.zeros(shape=(y_valid.shape[0],))
    epoch_val_loss = np.zeros(shape=(y_valid.shape[0],))
    epoch_val_mae = np.zeros(shape=(y_valid.shape[0],))
    epoch_val_rmse = np.zeros(shape=(y_valid.shape[0],))
    epoch_val_rsquare = np.zeros(shape=(y_valid.shape[0],))
    
    for i in trange(0, len(y_valid)):
        with tf.GradientTape() as tape:
            y_pred = lstm_model(x_valid[i:i+1,:,:], training=False)
        MAE.update_state(y_valid[i], y_pred)
        RMSE.update_state(y_valid[i], y_pred)
        RSquare.update_state(y_valid[i:i+1], np.array(y_pred[0]))
        PRED[i] = y_pred
        epoch_val_loss[i] = custom_loss(y_valid[i], y_pred)
        epoch_val_mae[i] = MAE.result()
        epoch_val_rmse[i] = RMSE.result()
        epoch_val_rsquare[i] = RSquare.result()
        
    print(f'val_loss: {np.mean(epoch_val_loss)}')
    print(f'val_mae: {np.mean(epoch_val_mae)}')
    print(f'val_rmse: {np.mean(epoch_val_rmse)}')
    print(f'val_rsquare: {np.mean(epoch_val_rsquare)}')

    lstm_model.save(filepath=f'{model_loc}{iEpoch}',
                        overwrite=True, include_optimizer=True,
                        save_format='h5', signatures=None, options=None,
                        save_traces=True
                )
    prev_model = tf.keras.models.clone_model(lstm_model)

    if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
        os.remove(f'{model_loc}{iEpoch-1}.ch')
    os.mknod(f'{model_loc}{iEpoch}.ch')

    # Saving history variable
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(data={
            'loss' : [loss_value],
            'val_loss' : [np.mean(epoch_val_loss)],
            'val_mean_absolute_error' : [np.mean(epoch_val_mae)],
            'val_root_mean_squared_error' : [np.mean(epoch_val_rmse)],
            'val_r_square' : [np.mean(epoch_val_rsquare)],
        })
    # or save to csv:
    with open(f'{model_loc}history-{profile}.csv', mode='a') as f:
        if(firstLog):
            hist_df.to_csv(f, index=False)
            firstLog = False
        else:
            hist_df.to_csv(f, index=False, header=False)
    
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_valid[::,]-PRED)))
    PERF = np.array([np.mean(epoch_val_loss), np.mean(epoch_val_mae),
                     np.mean(epoch_val_rmse), np.mean(epoch_val_rsquare)],
                     dtype=np.float32)
    # otherwise the right y-label is slightly clipped
    predicting_plot(profile=profile, file_name='Model №7',
                    model_loc=model_loc,
                    model_type='LSTM Test - Train dataset',
                    iEpoch=f'val-{iEpoch}',
                    Y=y_valid,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=PERF,
                    TAIL=y_valid.shape[0],
                    save_plot=True,
                    RMS_plot=False)
    if(PERF[-2] <=0.024): # Check thr RMSE
        print("RMS droped around 2.4%. Breaking the training")
        break
# %%