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

# Compared with an LST<-based RNN model,a GRU-based RNN model has a simpler 
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
import datetime
import logging
import os, sys, getopt    # OS, SYS, argc functions
from sys import platform  # Get type of OS

import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
# plt.switch_backend('agg')       #! FIX in the no-X env: RuntimeError: Invalid DISPLAY variable
import numpy as np
import pandas as pd  # File read
import tensorflow as tf  # Tensorflow and Numpy replacement
import tensorflow_addons as tfa
from tqdm import tqdm, trange

from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
from py_modules.tf_modules import scheduler, get_learning_rate
from py_modules.utils import str2bool, Locate_Best_Epoch
from py_modules.plotting import predicting_plot, history_plot

import sys
from typing import Callable
if (sys.version_info[1] < 9):
    LIST = list
    from typing import List as list
    from typing import Tuple as tuple

# %%
# Extract params
# try:
#     opts, args = getopt.getopt(sys.argv[1:],"hd:e:l:n:a:g:p:",
#                     ["help", "debug=", "epochs=", "layers=", "neurons=",
#                      "attempt=", "gpu=", "profile="])
# except getopt.error as err: 
#     # output error, and return with an error code 
#     print (str(err)) 
#     print ('EXEPTION: Arguments requied!')
#     sys.exit(2)

opts = [('-d', 'False'), ('-e', '100'), ('-l', '3'), ('-n', '131'), ('-a', '11'),
        ('-g', '0'), ('-p', 'FUDS')] # 2x131 1x1572 
debug   : int = 0
batch   : int = 1
mEpoch  : int = 10
nLayers : int = 1
nNeurons: int = 262
attempt : str = '1'
GPU     : int = None
profile : str = 'DST'
rounding: int = 5
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
            debug = 1
        else:
            logging.basicConfig(level=logging.CRITICAL)
            logging.warning("Logger Critical")
            debug = 0
    elif opt in ("-e", "--epochs"):
        mEpoch = int(arg)
    elif opt in ("-l", "--layers"):
        nLayers = int(arg)
    elif opt in ("-n", "--neurons"):
        nNeurons = int(arg)
    elif opt in ("-a", "--attempts"):
        attempt = (arg)
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
Data : str = ''
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
                              PROFILE_range = profile,
                              round=rounding)
# %%
window = WindowGenerator(Data=dataGenerator,
                        input_width=500, label_width=1, shift=0,
                        input_columns=['Current(A)', 'Voltage(V)',
                                                'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=batch,
                        includeTarget=False, normaliseLabal=False,
                        shuffleTraining=False,
                        normaliseInput=True,
                        round=rounding)
x_train, y_train = window.train
x_valid, y_valid = window.valid
x_testi, y_testi = window.test

tv_length = len(x_valid)
xt_valid = np.array(x_train[-tv_length:,:,:], copy=True, dtype=np.float32)
yt_valid = np.array(y_train[-tv_length:,:]  , copy=True, dtype=np.float32)
# %%
def create_model(mFunc : Callable, layers : int = 1,
                 neurons : int = 500, dropout : float = 0.2,
                 input_shape : tuple = (500, 3), batch : int = 1
            ) -> tf.keras.models.Sequential:
    """ Creates Tensorflow 2 based time series models with inputs exception 
    handeling. Accepts multilayer models.
    TODO: For Py3.10 correct typing with: func( .. dropout : int | float .. )

    Args:
        mFunc (Callable): Time series model function. .LSTM or .GRU
        layers (int, optional): № of layers. Above 1 will create a return
        sequence based models. Defaults to 1.
        neurons (int, optional): Total № of neurons across all layers.
        Value will be splitted evenly across all layers floored with 
        int() function. Defaults to 500.
        dropout (float, optional): Percentage dropout to eliminate random
        values. Defaults to 0.2.
        input_shape (tuple, optional): Input layer shape typle. Describes:
        (№_of_samles, №_of_deatues). Defaults to (500, 3).
        batch (int, optional): Batch size used at input layer. Defaults to 1.

    Raises:
        ZeroDivisionError: Rise an exception of anticipates unhandeled layer
        value, which cannot split neurons.

    Returns:
        tf.keras.models.Sequential: A sequentil model with single output and
        sigmoind() as an activation function.
    """
    # Check layers, neurons, dropout and batch are acceptable
    layers = 1 if layers == 0 else abs(layers)
    units : int = int(500/layers) if neurons == 0 else int(abs(neurons)/layers)
    dropout : float = float(dropout) if dropout >= 0 else float(abs(dropout))
    #? int(batch) if batch > 0 else ( int(abs(batch)) if batch != 0 else 1 )
    batch : int = int(abs(batch)) if batch != 0 else 1
    
    # Define sequential model with an Input Layer
    model : tf.keras.models.Sequential = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape,
                                batch_size=1)
        ])
    
    # Fill the layer content
    if(layers > 1): #* Middle connection layers
        for _ in range(layers-1):
            model.add(mFunc(
                    units=units, activation='tanh',
                    dropout=dropout, return_sequences=True
                ))
    if(layers > 0): #* Last no-connection layer
        model.add(mFunc(
                units=units, activation='tanh',
                dropout=dropout, return_sequences=False
            ))
    else:
        print("Unhaldeled exeption with Layers")
        raise ZeroDivisionError
    
    # Define the last Output layer with sigmoind
    model.add(tf.keras.layers.Dense(
            units=1, activation='sigmoid', use_bias=True
        ))
    
    # Return completed model with some info if neededs
    # print(model.summary())
    return model
    
custom_loss = lambda y_true, y_pred: tf.keras.backend.mean(
            x=tf.math.squared_difference(
                    x=tf.cast(x=y_true, dtype=y_pred.dtype),
                    y=tf.convert_to_tensor(value=y_pred)
                ),
            axis=-1,
            keepdims=False
        )

file_name : str = os.path.basename(__file__)[:-3]
model_name : str = 'ModelsUp-2'
####################! ADD model_name to path!!! ################################
model_loc : str = f'Mods/{model_name}/{nLayers}x{file_name}-({nNeurons})/{attempt}-{profile}/'
firstLog : bool = True
iLr     : float = 0.001
prev_error : np.float32 = 1.0

iEpoch : int = 0
p2     : int = int(mEpoch/3)
skipCompile1, skipCompile2 = False, False
try:
    iEpoch, prev_error  = Locate_Best_Epoch(f'{model_loc}history.csv', 'mae')
    lstm_model : tf.keras.models.Sequential = tf.keras.models.load_model(
            f'{model_loc}{iEpoch}',
            compile=False)
    iLr = get_learning_rate(iEpoch, iLr, 'linear')
    firstLog = False
    print(f"Model Identefied at {iEpoch} with {prev_error}. Continue training.")
except OSError as identifier:
    print("Model Not Found, creating new. {} \n".format(identifier))
    # gru_model = tf.keras.models.Sequential([
    #     tf.keras.layers.InputLayer(input_shape=x_train.shape[-2:],
    #                                batch_size=None),
    #     tf.keras.layers.GRU(    #?260 by BinXia, times by 2 or 3
    #         units=560, activation='tanh', recurrent_activation='sigmoid',
    #         use_bias=True, kernel_initializer='glorot_uniform',
    #         recurrent_initializer='orthogonal', bias_initializer='zeros',
    #         kernel_regularizer=None,
    #         recurrent_regularizer=None, bias_regularizer=None,
    #         activity_regularizer=None, kernel_constraint=None,
    #         recurrent_constraint=None, bias_constraint=None, dropout=0.2,
    #         recurrent_dropout=0.0, return_sequences=False, return_state=False,
    #         go_backwards=False, stateful=False, unroll=False, time_major=False,
    #         reset_after=True
    #     ),
    #     tf.keras.layers.Dense(units=1,
    #                           activation='sigmoid')
    # ])
    if type(x_train) == list:
        input_shape : tuple = x_train[0].shape[-2:]
    else:
        input_shape : tuple = x_train.shape[-2:]
    gru_model = create_model(
            tf.keras.layers.GRU, layers=nLayers, neurons=nNeurons,
            dropout=0.2, input_shape=input_shape, batch=1
        )
    iLr = 0.001
    firstLog = True
prev_model = tf.keras.models.clone_model(gru_model)

# checkpoints = tf.keras.callbacks.ModelCheckpoint(
#     filepath =model_loc+f'{profile}-checkpoints/checkpoint',
#     monitor='val_loss', verbose=0,
#     save_best_only=False, save_weights_only=False,
#     mode='auto', save_freq='epoch', options=None,
# )

# tensorboard_callback = tf.keras.callbacks.TensorBoard(
#         log_dir=model_loc+
#             f'tensorboard/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
#         histogram_freq=1, write_graph=True, write_images=False,
#         update_freq='epoch', profile_batch=2, embeddings_freq=0,
#         embeddings_metadata=None
#     )

nanTerminate = tf.keras.callbacks.TerminateOnNaN()
# %%
if not os.path.exists(f'{model_loc}'):
    os.makedirs(f'{model_loc}')
if not os.path.exists(f'{model_loc}traiPlots'):
    os.mkdir(f'{model_loc}traiPlots')
if not os.path.exists(f'{model_loc}valdPlots'):
    os.mkdir(f'{model_loc}valdPlots')
if not os.path.exists(f'{model_loc}testPlots'):
    os.mkdir(f'{model_loc}testPlots')
if not os.path.exists(f'{model_loc}history.csv'):
    #! ADD EPOCH FOR FUTURE, THEN FIX THE BEST EPOCH
    print("History not created. Making")
    with open(f'{model_loc}history.csv', mode='w') as f:
        f.write('Epoch,loss,mae,rmse,rsquare,time(s),'
                'train_l,train_mae,train_rms,train_r_s,'
                'vall_l,val_mae,val_rms,val_r_s,val_t_s,'
                'test_l,tes_mae,tes_rms,tes_r_s,tes_t_s,learn_r\n')
if not os.path.exists(f'{model_loc}history-cycles.csv'):
    print("Cycle-History not created. Making")
    with open(f'{model_loc}history-cycles.csv', mode='w') as f:
        f.write('Epoch,Cycle,'
                'train_l,train_mae,train_rms,train_t_s,'
                'vall_l,val_mae,val_rms,val_t_s,'
                'learn_r\n')
if not os.path.exists(f'{model_loc}cycles-log'):
    os.mkdir(f'{model_loc}cycles-log')

#* Save the valid logits to separate files. Just annoying
pd.DataFrame(y_train[:,0,0]).to_csv(f'{model_loc}y_train.csv', sep = ",", na_rep = "", line_terminator = '\n')
pd.DataFrame(yt_valid[:,0,0]).to_csv(f'{model_loc}yt_valid.csv', sep = ",", na_rep = "", line_terminator = '\n')
pd.DataFrame(y_valid[:,0,0]).to_csv(f'{model_loc}y_valid.csv', sep = ",", na_rep = "", line_terminator = '\n')
pd.DataFrame(y_testi[:,0,0]).to_csv(f'{model_loc}y_testi.csv', sep = ",", na_rep = "", line_terminator = '\n')

i_attempts : int = 0
n_attempts : int = 50
skip       : int = 1
firtstEpoch: bool = True
while iEpoch < mEpoch:
    if (iEpoch<=p2 and not skipCompile1):
        gru_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Nadam(learning_rate=iLr,
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
                optimizer=tf.keras.optimizers.Adamax(learning_rate=iLr,
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
    tic : float = time.perf_counter()
    history = gru_model.fit(x=x_train[:,0,:,:], y=y_train[:,0,:], epochs=1,
                        validation_data=(x_valid[:,0,:,:], y_valid[:,0,:]),
                        callbacks=[nanTerminate],
                        batch_size=1, shuffle=True
                        )
    toc : float = time.perf_counter() - tic
    # pbar.close()
    cLr = gru_model.optimiser.lr
    print(f'Epoch {iEpoch}/{mEpoch} :: '
            f'Elapsed Time: {toc} - '
            # f'loss: {loss_value[0]:.4f} - '
            f'mae: {MAE.result():.4f} - '
            f'rmse: {RMSE.result():.4f} - '
            f'rsquare: {RSquare.result():.4f} - '
            f'Lear-Rate: {cLr} - '
        )
    #? Dealing with NaN state. Give few trials to see if model improves
    if (tf.math.is_nan(history.history['loss'])):
        print('NaN model')
        while i_attempts < n_attempts:
            print(f'Attempt {i_attempts}')
            gru_model = tf.keras.models.clone_model(prev_model)
            #! Single compiler selection
            gru_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0005,
                        beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Adamax'
                        ),
                    metrics=[tf.metrics.MeanAbsoluteError(),
                            tf.metrics.RootMeanSquaredError(),
                            tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
                )
            history = gru_model.fit(x=x_train[:,:,:], y=y_train[:,:], epochs=1,
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
            gru_model.save(filepath=f'{model_loc}{iEpoch}-{i_attempts}',
                            overwrite=True, include_optimizer=True,
                            save_format='h5', signatures=None, options=None,
                            save_traces=True
                )
            # gru_model.save_weights(f'{model_loc}weights/{iEpoch}-{i_attempts}')
            i_attempts = 0
            prev_model = tf.keras.models.clone_model(gru_model)
    else:
        gru_model.save(filepath=f'{model_loc}{iEpoch}',
                       overwrite=True, include_optimizer=True,
                       save_format='h5', signatures=None, options=None,
                       save_traces=True
                )
        # gru_model.save_weights(f'{model_loc}weights/{iEpoch}')
        prev_model = tf.keras.models.clone_model(gru_model)
    
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
    PRED = gru_model.predict(x_valid,batch_size=1)
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_valid[::skip,]-PRED)))
    PERF = gru_model.evaluate(x=x_valid,
                              y=y_valid,
                              batch_size=1,
                              verbose=0)
    # otherwise the right y-label is slightly clipped
    predicting_plot(profile=profile, file_name='Model №2',
                    model_loc=model_loc,
                    model_type='GRU Train',
                    iEpoch=f'val-{iEpoch}',
                    Y=y_valid,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=PERF,
                    TAIL=y_valid.shape[0],
                    save_plot=True)
    if(PERF[-2] <=0.024): # Check thr RMSE
        print("RMS droped around 2.4%. Breaking the training")
        break

# %%
PRED = gru_model.predict(x_test_one, batch_size=1, verbose=1)
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(y_test_one[::,]-PRED)))
if profile == 'DST':
    predicting_plot(profile=profile, file_name='Model №2',
                    model_loc=model_loc,
                    model_type='GRU Test on US06', iEpoch=f'Test One-{iEpoch}',
                    Y=y_test_one,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=gru_model.evaluate(
                                    x=x_test_one,
                                    y=y_test_one,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_one.shape[0],
                    save_plot=True)
else:
    predicting_plot(profile=profile, file_name='Model №2',
                    model_loc=model_loc,
                    model_type='GRU Test on DST', iEpoch=f'Test One-{iEpoch}',
                    Y=y_test_one,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=gru_model.evaluate(
                                    x=x_test_one,
                                    y=y_test_one,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_one.shape[0],
                    save_plot=True)

PRED = gru_model.predict(x_test_two, batch_size=1, verbose=1)
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(y_test_two[::,]-PRED)))
if profile == 'FUDS':
    predicting_plot(profile=profile, file_name='Model №2',
                    model_loc=model_loc,
                    model_type='GRU Test on US06', iEpoch=f'Test Two-{iEpoch}',
                    Y=y_test_two,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=gru_model.evaluate(
                                    x=x_test_two,
                                    y=y_test_two,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_two.shape[0],
                    save_plot=True)
else:
    predicting_plot(profile=profile, file_name='Model №2',
                    model_loc=model_loc,
                    model_type='GRU Test on FUDS', iEpoch=f'Test Two-{iEpoch}',
                    Y=y_test_two,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=gru_model.evaluate(
                                    x=x_test_two,
                                    y=y_test_two,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_two.shape[0],
                    save_plot=True)
# %%
# Convert the model to Tensorflow Lite and save.
with open(f'{model_loc}Model-№2-{profile}.tflite', 'wb') as f:
    f.write(
        tf.lite.TFLiteConverter.from_keras_model(
                model=gru_model
            ).convert()
        )
