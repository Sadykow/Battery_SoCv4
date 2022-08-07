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
try:
    opts, args = getopt.getopt(sys.argv[1:],"hd:e:l:n:a:g:p:o:r:",
                    ["help", "debug=", "epochs=", "layers=", "neurons=",
                     "attempt=", "gpu=", "profile=", "optimiser=", "rate="])
except getopt.error as err: 
    # output error, and return with an error code 
    print (str(err)) 
    print ('EXEPTION: Arguments requied!')
    sys.exit(2)

# opts = [('-d', 'False'), ('-e', '100'), ('-l', '3'), ('-n', '131'), ('-a', '1'),
#         ('-g', '0'), ('-p', 'FUDS'), ('-o', 'AdaMax'), ('-r', '0.0001')] # 2x131 1x1572 
debug   : int = 0
batch   : int = 1
mEpoch  : int = 10
nLayers : int = 1
nNeurons: int = 262
attempt : str = '1'
GPU     : int = None
profile : str = 'DST'
rounding: int = 5
optimiser_type = "Adam"
rate    = 0.001
print(opts)
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
    elif opt in ("-o", "--optimiser"):
        optimiser_type = (arg)
    elif opt in ("-r", "--rate"):
        rate = float(arg)
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
    # for device in physical_devices:
    #     tf.config.experimental.set_memory_growth(
    #                         device=device, enable=True)
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
    
# custom_loss = lambda y_true, y_pred: tf.keras.backend.mean(
#             x=tf.math.squared_difference(
#                     x=tf.cast(x=y_true, dtype=y_pred.dtype),
#                     y=tf.convert_to_tensor(value=y_pred)
#                 ),
#             axis=-1,
#             keepdims=False
#         )

file_name : str = "BinXiao2020" #os.path.basename(__file__)[:-3]
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
    gru_model : tf.keras.models.Sequential = tf.keras.models.load_model(
            f'{model_loc}{iEpoch}',
            compile=False)
    firstLog = False
    if (optimiser_type == 'AdaMax'):
        iLr = rate
        optimiser = tf.keras.optimizers.Adamax(learning_rate=iLr,
                        beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Adamax'
                    )
        constLearning = True
        n_attempts : int = 5
        print('>>> Using Adamax for Fune-Tuning')
    else:
        iLr = get_learning_rate(iEpoch, iLr, 'linear')
        optimiser = tf.keras.optimizers.Nadam(learning_rate=iLr,
                    beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Nadam'
                )
        constLearning = False
        n_attempts : int = 50
        print('>>> Using Nadam for Pre-Tuning')
    print(f"Model Identefied at {iEpoch} with {prev_error}.")
    #! I must find a way to make use of it
except (OSError, TypeError) as identifier:
    print("Model Not Found, creating new with Nadam. {} \n".format(identifier))
    # gru_model = tf.keras.models.Sequential([
    #     tf.keras.layers.InputLayer(input_shape=x_train.shape[-2:],
    #                                batch_size=None),
    #     tf.keras.layers.GRU(    #?260 by BinXia, times by 2 or 3
    #         units=560, activation='tanh', recurrent_activation='sigmoid',
    #         use_bias=True, kernel_initializer='glorot_uniform',
    #         recurrent_initializer='orthogonal', bias_initializer='zeros',
    #         kernel_regularizer=None,
    #         recurrent_regularizer=None, bias_regularizer=NPone,
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
    optimiser = tf.keras.optimizers.Nadam(learning_rate=iLr,
                    beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Nadam'
                )
    constLearning = False
    n_attempts : int = 50
    iLr = 0.001
    firstLog = True
prev_model = tf.keras.models.clone_model(gru_model)

# %%
loss_fn   = tf.losses.MeanAbsoluteError(
                    reduction=tf.keras.losses.Reduction.NONE,
                )
MAE     = tf.metrics.MeanAbsoluteError()
RMSE    = tf.metrics.RootMeanSquaredError()
RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)

@tf.function
def train_single_st(input : tuple[np.ndarray, np.ndarray],
                    metrics : tf.keras.metrics,
                    ) -> tf.Tensor:
    # Execute model as training
    with tf.GradientTape() as tape:
        #? tf.EagerTensor['1,1', tf.float32]
        logits     : tf.Tensor = gru_model(input[0], training=True)
        #? tf.EagerTensor['1,', tf.float32]
        loss_value : tf.Tensor = loss_fn(input[1], logits)
    
    # Get gradients and apply optimiser to model
    grads : list[tf.Tensor] = tape.gradient(
                    loss_value,
                    gru_model.trainable_weights
                )
    optimiser.apply_gradients(zip(grads, gru_model.trainable_weights))
    
    # Update metrics before
    for metric in metrics:
        metric.update_state(y_true=input[1], y_pred=logits)

    return loss_value

@tf.function
def test_step(input : tuple[np.ndarray, np.ndarray]) -> tf.Tensor:
    return gru_model(input, training=False)

def valid_loop(dist_input  : tuple[np.ndarray, np.ndarray],
               verbose : int = 0) -> tf.Tensor:
    x, y = dist_input
    logits  : np.ndarray = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
    loss    : np.ndarray = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
    val_MAE     = tf.metrics.MeanAbsoluteError()
    val_RMSE    = tf.metrics.RootMeanSquaredError()
    val_RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)
    
    # Debug verbose param
    if verbose == 1:
        rangeFunc : Callable = trange
    else:
        rangeFunc : Callable = range
    
    #! Prediction on this part can be paralylised across two GPUs. Like OpenMP
    for i in rangeFunc(y.shape[0]):
        logits[i] = test_step(x[i,:,:,:])
        val_MAE.update_state(y_true=y[i],     y_pred=logits[i])
        val_RMSE.update_state(y_true=y[i],    y_pred=logits[i])
        val_RSquare.update_state(y_true=y[i], y_pred=logits[i])
        
        loss[i] = loss_fn(y[i], logits[i])

    #! Error with RMSE here. No mean should be used.
    mae      : float = val_MAE.result()
    rmse     : float = val_RMSE.result()
    r_Square : float = val_RSquare.result()
    
    # Reset training metrics at the end of each epoch
    val_MAE.reset_states()
    val_RMSE.reset_states()
    val_RSquare.reset_states()

    return [loss, mae, rmse, r_Square, logits]

def fast_valid(dist_input  : tuple[np.ndarray, np.ndarray],
               verbose : int = 0) -> tf.Tensor:
    x, y = dist_input
    logits  : np.ndarray = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
    loss    : np.ndarray = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
    val_MAE     = tf.metrics.MeanAbsoluteError()
    val_RMSE    = tf.metrics.RootMeanSquaredError()

    # Debug verbose param
    if verbose == 1:
        rangeFunc : Callable = trange
    else:
        rangeFunc : Callable = range
    
    #! Prediction on this part can be paralylised across two GPUs. Like OpenMP
    tic : float = time.perf_counter()
    for i in rangeFunc(y.shape[0]):
        logits[i] = test_step(x[i,:,:,:])
        val_MAE.update_state(y_true=y[i],     y_pred=logits[i])
        val_RMSE.update_state(y_true=y[i],    y_pred=logits[i])
        
        loss[i] = loss_fn(y[i], logits[i])

    toc : float = time.perf_counter() - tic
    mae      : float = val_MAE.result()
    rmse     : float = val_RMSE.result()

    # Reset training metrics at the end of each epoch
    val_MAE.reset_states()
    val_RMSE.reset_states()

    return [loss, mae, rmse, toc, logits]

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
# n_attempts : int = 50 # Realocated to the initialisation
skip       : int = 1
firtstEpoch: bool = True
while iEpoch < mEpoch:
    iEpoch+=1
    print(f"Epoch {iEpoch}/{mEpoch}")
    # pbar = tqdm(total=y_train.shape[0])
    tic : float = time.perf_counter()
    sh_i = np.arange(y_train.shape[0])
    np.random.shuffle(sh_i)
    print(f'Commincing Epoch: {iEpoch}')
    loss_value : np.float32 = [0.0]
    for i in sh_i[::]:
        loss_value = train_single_st((x_train[i,:,:,:], y_train[i,:]),
                                    metrics=[MAE,RMSE,RSquare],
                                    #curr_error=CR_ER
                                    )
        # Progress Bar
        # pbar.update(1)
        # pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
        #                         # f'loss: {(loss_value[0]):.4f} - '
        #                         f'mae: {MAE.result():.4f} - '
        #                         f'rmse: {RMSE.result():.4f} - '
        #                         # f'rsquare: {RSquare.result():.4f} --- '
        #                     )
    toc : float = time.perf_counter() - tic
    # pbar.close()
    cLr = optimiser.get_config()['learning_rate']
    print(f'Epoch {iEpoch}/{mEpoch} :: '
            f'Elapsed Time: {toc} - '
            # f'loss: {loss_value[0]:.4f} - '
            f'mae: {MAE.result():.4f} - '
            f'rmse: {RMSE.result():.4f} - '
            f'rsquare: {RSquare.result():.4f} - '
            f'Lear-Rate: {cLr} - '
        )
#?==================                ===========================================
    #* Dealing with NaN state. Give few trials to see if model improves
    curr_error = MAE.result().numpy()
    print(f'The post optimiser error: {curr_error}', flush=True)
    if (tf.math.is_nan(loss_value[0]) or curr_error > prev_error):
        print('->> NaN or High error model')
        i_attempts : int = 0
        firstFaltyLog : bool = True
        while i_attempts < n_attempts:
            print(f'->>> Attempt {i_attempts}')
            try:
                gru_model.save(filepath=f'{model_loc}{iEpoch}-fail-{i_attempts}',
                        overwrite=True, include_optimizer=True,
                        save_format='h5', signatures=None, options=None
                )
            except OSError:
                os.remove(f'{model_loc}{iEpoch}-fail-{i_attempts}')
                gru_model.save(filepath=f'{model_loc}{iEpoch}-fail-{i_attempts}',
                        overwrite=True, include_optimizer=True,
                        save_format='h5', signatures=None, options=None
                )
            gru_model = tf.keras.models.clone_model(prev_model)
            
            np.random.shuffle(sh_i)
            # pbar = tqdm(total=y_train.shape[0])

            # Reset every metric
            MAE.reset_states()
            RMSE.reset_states()
            RSquare.reset_states()

            tic = time.perf_counter()
            for i in sh_i[::]:
                loss_value = train_single_st(
                                        (x_train[i,:,:,:], y_train[i,:]),
                                        [MAE,RMSE,RSquare],
                                        # curr_error=CR_ER
                                        )
                # Progress Bar
                # pbar.update(1)
                # pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
                #                     f'loss: {(loss_value[0]):.4f} - '
                #                     )
            toc = time.perf_counter() - tic
            # pbar.close()
            TRAIN = valid_loop((xt_valid, yt_valid), verbose = debug)
            
            # Update learning rate
            if(not constLearning):
                iLr /= 2
                optimiser.learning_rate = iLr
                print(f">>> Updating iLR with {iLr}")
            else:
                print(f">>> Keeping iLR at {iLr}")

            # Log the faulty results
            faulty_hist_df = pd.DataFrame(data={
                    'Epoch'  : [iEpoch],
                    'attempt': [i_attempts],
                    'loss'   : [np.array(loss_value[0])],
                    'mae'    : [np.array(MAE.result())],
                    'time(s)': [np.array(toc)],
                    'learning_rate' : [np.array(iLr)],
                    'train_l' : np.mean(TRAIN[0]),
                    'train_mae': np.array(TRAIN[1]),
                    'train_rms': np.array(TRAIN[2]),
                    'train_r_s': np.array(TRAIN[3]),
                })
            with open(f'{model_loc}{iEpoch}-faulty-history.csv',
                        mode='a') as f:
                if(firstFaltyLog):
                    faulty_hist_df.to_csv(f, index=False)
                    firstFaltyLog = False
                else:
                    faulty_hist_df.to_csv(f, index=False, header=False)
            curr_error = MAE.result().numpy()
            print(
                f'The post optimiser error: {curr_error}'
                f'with L-rate {optimiser.get_config()["learning_rate"]}'
                )
            if (not tf.math.is_nan(loss_value[0]) and
                not curr_error > prev_error and
                not TRAIN[1] > 0.20 ):
                print(f'->>> Attempt {i_attempts} Passed')
                break
            else:
                i_attempts += 1
        if (i_attempts == n_attempts):
                # and (tf.math.is_nan(history.history['loss'])):
            print('->> Model reached the optimum -- Breaking')
            break
        else:
            print('->> Model restored -- continue training')
            gru_model.save(filepath=f'{model_loc}{iEpoch}',
                            overwrite=True, include_optimizer=True,
                            save_format='h5', signatures=None, options=None
                )
            prev_model = tf.keras.models.clone_model(gru_model)
            prev_error = curr_error
    else:
        gru_model.save(filepath=f'{model_loc}{iEpoch}',
                       overwrite=True, include_optimizer=True,
                       save_format='h5', signatures=None, options=None
                )
        prev_model = tf.keras.models.clone_model(gru_model)
        prev_error = curr_error
    
    # Update learning rate
    if (constLearning):
        optimiser.learning_rate = iLr = rate
    else:
        iLr = scheduler(iEpoch, iLr, 'linear')
        optimiser.learning_rate = iLr
    
    # Validating trained model 
    TRAIN = valid_loop((xt_valid, yt_valid), verbose = debug)
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                yt_valid[:,0,0]-TRAIN[4])))
    predicting_plot(profile=profile, file_name=model_name,
                    model_loc=f'{model_loc}/traiPlots/',
                    model_type='LSTM valid',
                    iEpoch=f'tra-{iEpoch}',
                    Y=yt_valid[:,0],
                    PRED=TRAIN[4],
                    RMS=RMS,
                    val_perf=[np.mean(TRAIN[0]), TRAIN[1],
                            TRAIN[2], TRAIN[3]],
                    TAIL=yt_valid.shape[0],
                    save_plot=True)
    print(f'Epoch {iEpoch}/{mEpoch} :: TRAIN :: '
            f'mae: {TRAIN[1]:.4f} - '
            f'rmse: {TRAIN[2]:.4f} - '
            f'rsquare: {TRAIN[3]:.4f} - '
            f'\n'
        )
    # Validating model 
    val_tic : float = time.perf_counter()
    PERF = valid_loop((x_valid, y_valid), verbose = debug)
    # PERF = valid_loop((x_train, y_train), verbose = debug)
    val_toc : float = time.perf_counter() - val_tic
    #! Verefy RMS shape
    #! if RMS.shape[0] == RMS.shape[1]
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_valid[:,0,0]-PERF[4])))
    predicting_plot(profile=profile, file_name=model_name,
                    model_loc=f'{model_loc}/valdPlots/',
                    model_type='LSTM valid',
                    iEpoch=f'val-{iEpoch}',
                    Y=y_valid[:,0],
                    PRED=PERF[4],
                    RMS=RMS,
                    val_perf=[np.mean(PERF[0]), PERF[1],
                            PERF[2], PERF[3]],
                    TAIL=y_valid.shape[0],
                    save_plot=True)
    print(f'Epoch {iEpoch}/{mEpoch} :: PERF :: '
            f'Elapsed Time: {val_toc} - '
            f'mae: {PERF[1]:.4f} - '
            f'rmse: {PERF[2]:.4f} - '
            f'rsquare: {PERF[3]:.4f} - '
            f'\n'
        )
    # Testing model 
    mid_one = int(x_testi.shape[0]/2)#+350
    mid_two = int(x_testi.shape[0]/2)+400
    ts_tic : float = time.perf_counter()
    TEST1 = valid_loop((x_testi[:mid_one], y_testi[:mid_one]), verbose = debug)
    TEST2 = valid_loop((x_testi[mid_two:], y_testi[mid_two:]), verbose = debug)
    ts_toc : float = time.perf_counter() - ts_tic
    #! Verefy RMS shape
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_testi[:mid_one,0,0]-TEST1[4])))
    #! If statement for string to change
    if profile == 'DST':
        save_title_type : str = 'GRU Test on US06'
        save_file_name  : str = f'US06-{iEpoch}'
    else:
        save_title_type : str = 'GRU Test on DST'
        save_file_name  : str = f'DST-{iEpoch}'

    predicting_plot(profile=profile, file_name=model_name,
                    model_loc=f'{model_loc}/testPlots/',
                    model_type=save_title_type,
                    iEpoch=save_file_name,
                    Y=y_testi[:mid_one,0],
                    PRED=TEST1[4],
                    RMS=RMS,
                    val_perf=[np.mean(TEST1[0]), TEST1[1],
                            TEST1[2], TEST1[3]],
                    TAIL=y_testi.shape[0],
                    save_plot=True)

    if profile == 'FUDS':
        save_title_type : str = 'GRU Test on US06'
        save_file_name  : str = f'US06-{iEpoch}'
    else:
        save_title_type : str = 'GRY Test on FUDS'
        save_file_name  : str = f'FUDS-{iEpoch}'
    #! Verefy RMS shape
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_testi[mid_two:,0,0]-TEST2[4])))

    predicting_plot(profile=profile, file_name=model_name,
                    model_loc=f'{model_loc}/testPlots/',
                    model_type=save_title_type,
                    iEpoch=save_file_name,
                    Y=y_testi[mid_two:,0],
                    PRED=TEST2[4],
                    RMS=RMS,
                    val_perf=[np.mean(TEST2[0]), TEST2[1],
                            TEST2[2], TEST2[3]],
                    TAIL=y_testi.shape[0],
                    save_plot=True)
    print(f'Epoch {iEpoch}/{mEpoch} :: TEST :: '
            f'Elapsed Time: {ts_toc} - '
            f'mae: {np.mean(np.append(TEST1[1], TEST2[1])):.4f} - '
            f'rmse: {np.mean(np.append(TEST1[2], TEST2[2])):.4f} - '
            f'rsquare: {np.mean(np.append(TEST1[3], TEST2[3])):.4f} - '
            f'\n'
        )

    # Saving history variable
    hist_df : pd.DataFrame = pd.read_csv(f'{model_loc}history.csv',
                                            index_col='Epoch')
    hist_df = hist_df.reset_index()

    #! Rewrite as add, not a new, similar to the one I found on web with data analysis
    hist_ser = pd.Series(data={
            'Epoch'  : iEpoch,
            'loss'   : np.array(loss_value[0]),
            'mae'    : np.array(MAE.result()),
            'rmse'   : np.array(RMSE.result()),
            'rsquare': np.array(RSquare.result()),
            'time(s)': toc,
            'train_l' : np.mean(TRAIN[0]),
            'train_mae': np.array(TRAIN[1]),
            'train_rms': np.array(TRAIN[2]),
            'train_r_s': np.array(TRAIN[3]),
            'vall_l' : np.mean(PERF[0]),
            'val_mae': np.array(PERF[1]),
            'val_rms': np.array(PERF[2]),
            'val_r_s': np.array(PERF[3]),
            'val_t_s': val_toc,
            'test_l' : np.mean(np.append(TEST1[0], TEST2[0])),
            'tes_mae': np.mean(np.append(TEST1[1], TEST2[1])),
            'tes_rms': np.mean(np.append(TEST1[2], TEST2[2])),
            'tes_r_s': np.mean(np.append(TEST1[3], TEST2[3])),
            'tes_t_s': ts_toc,
            'learn_r': np.array(iLr)
        })
    if(len(hist_df[hist_df['Epoch']==iEpoch]) == 0):
        # hist_df = pd.concat([hist_df, hist_ser], ignore_index=True)
        hist_df = hist_df.append(hist_ser, ignore_index=True)
        # hist_df.loc[hist_df['Epoch']==iEpoch] = hist_ser
    else:
        hist_df.loc[len(hist_df)] = hist_ser
    hist_df.to_csv(f'{model_loc}history.csv', index=False, sep = ",", na_rep = "", line_terminator = '\n')

    # Plot History for reference and overwrite if have to    
    history_plot(profile, model_name, model_loc, hist_df, save_plot=True,
                    plot_file_name=f'history-{profile}-train.svg')
    history_plot(profile, model_name, model_loc, hist_df, save_plot=True,
                metrics=['mae', 'val_mae',
                        'rmse', 'val_rms'],
                plot_file_name=f'history-{profile}-valid.svg')

    pd.DataFrame(TRAIN[4]).to_csv(f'{model_loc}{iEpoch}-train-logits.csv')
    pd.DataFrame(PERF[4]).to_csv(f'{model_loc}{iEpoch}-valid-logits.csv')
    pd.DataFrame(np.append(TEST1[4], TEST2[4])
                ).to_csv(f'{model_loc}{iEpoch}-test--logits.csv')
    
    # Reset every metric and clear memory leak
    MAE.reset_states()
    RMSE.reset_states()
    RSquare.reset_states()

    # Flush and clean
    print('\n', flush=True)
    # tf.keras.backend.clear_session()

    #! Run the Evaluate function
    # PRED = gru_model.predict(x_valid,batch_size=1)
    # RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
    #             y_valid[::skip,]-PRED)))
    # PERF = gru_model.evaluate(x=x_valid,
    #                           y=y_valid,
    #                           batch_size=1,
    #                           verbose=0)
    # # otherwise the right y-label is slightly clipped
    # predicting_plot(profile=profile, file_name='Model №2',
    #                 model_loc=model_loc,
    #                 model_type='GRU Train',
    #                 iEpoch=f'val-{iEpoch}',
    #                 Y=y_valid,
    #                 PRED=PRED,
    #                 RMS=RMS,
    #                 val_perf=PERF,
    #                 TAIL=y_valid.shape[0],
    #                 save_plot=True)
    # if(PERF[-2] <=0.024): # Check thr RMSE
    #     print("RMS droped around 2.4%. Breaking the training")
    #     break
# %%
bestEpoch, _  = Locate_Best_Epoch(f'{model_loc}history.csv', 'mae')
gru_model : tf.keras.models.Sequential = tf.keras.models.load_model(
        f'{model_loc}{bestEpoch}',
        compile=False)
profiles: list = ['DST', 'US06', 'FUDS']
del dataGenerator
del window
dataGenerators : list = []
X : list = []
Y : list = []
for p in profiles:
    dataGenerators.append(
            DataGenerator(train_dir=f'{Data}A123_Matt_Set',
                          valid_dir=f'{Data}A123_Matt_Val',
                          test_dir=f'{Data}A123_Matt_Test',
                          columns=[
                            'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                            'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                          ],
                          PROFILE_range = p)
        )
for g in dataGenerators:
    _, x, y = WindowGenerator(Data=g,
                    input_width=500, label_width=1, shift=0,
                    input_columns=['Current(A)', 'Voltage(V)',
                                            'Temperature (C)_1'],
                    label_columns=['SoC(%)'], batch=1,
                    includeTarget=False, normaliseLabal=False,
                    shuffleTraining=False).train
    X.append(x)
    Y.append(y)
#! Run test against entire dataset to record the tabled accuracy
firstLog = False
for p, x, y in zip(profiles, X, Y):
    # Validating model 
    val_tic : float = time.perf_counter()
    PERF = valid_loop((x, y), verbose = debug)
    # PERF = gru_model.evaluate(x[:,0,:,:], y[:,0,:],
    #                         batch_size=1, verbose = debug)
    val_toc : float = time.perf_counter() - val_tic
    print(f'Profile {p} '
            f'Elapsed Time: {val_toc} - '
            f'mae: {PERF[1]:.4f} - '
            f'rmse: {PERF[2]:.4f} - '
            f'rsquare: {PERF[3]:.4f} - '
        )
        # Saving the log file
    hist_df = pd.DataFrame(data={
            'prof': [p],
            'loss': [np.mean(PERF[0])],
            'mae' : [np.array(PERF[1])],
            'rms' : [np.array(PERF[2])],
            'r2'  : [np.array(PERF[3])],
            't(s)': [val_toc],
        })
    with open(f'{model_loc}full-evaluation.csv', mode='a') as f:
        if(firstLog):
            hist_df.to_csv(f, index=False, header=True)
            firstLog = False
        else:
            hist_df.to_csv(f, index=False, header=False)

print('Model evaluation has been completed... finally....')
# %%
# PRED = gru_model.predict(x_test_one, batch_size=1, verbose=1)
# RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(y_test_one[::,]-PRED)))
# if profile == 'DST':
#     predicting_plot(profile=profile, file_name='Model №2',
#                     model_loc=model_loc,
#                     model_type='GRU Test on US06', iEpoch=f'Test One-{iEpoch}',
#                     Y=y_test_one,
#                     PRED=PRED,
#                     RMS=RMS,
#                     val_perf=gru_model.evaluate(
#                                     x=x_test_one,
#                                     y=y_test_one,
#                                     batch_size=1,
#                                     verbose=1),
#                     TAIL=y_test_one.shape[0],
#                     save_plot=True)
# else:
#     predicting_plot(profile=profile, file_name='Model №2',
#                     model_loc=model_loc,
#                     model_type='GRU Test on DST', iEpoch=f'Test One-{iEpoch}',
#                     Y=y_test_one,
#                     PRED=PRED,
#                     RMS=RMS,
#                     val_perf=gru_model.evaluate(
#                                     x=x_test_one,
#                                     y=y_test_one,
#                                     batch_size=1,
#                                     verbose=1),
#                     TAIL=y_test_one.shape[0],
#                     save_plot=True)

# PRED = gru_model.predict(x_test_two, batch_size=1, verbose=1)
# RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(y_test_two[::,]-PRED)))
# if profile == 'FUDS':
#     predicting_plot(profile=profile, file_name='Model №2',
#                     model_loc=model_loc,
#                     model_type='GRU Test on US06', iEpoch=f'Test Two-{iEpoch}',
#                     Y=y_test_two,
#                     PRED=PRED,
#                     RMS=RMS,
#                     val_perf=gru_model.evaluate(
#                                     x=x_test_two,
#                                     y=y_test_two,
#                                     batch_size=1,
#                                     verbose=1),
#                     TAIL=y_test_two.shape[0],
#                     save_plot=True)
# else:
#     predicting_plot(profile=profile, file_name='Model №2',
#                     model_loc=model_loc,
#                     model_type='GRU Test on FUDS', iEpoch=f'Test Two-{iEpoch}',
#                     Y=y_test_two,
#                     PRED=PRED,
#                     RMS=RMS,
#                     val_perf=gru_model.evaluate(
#                                     x=x_test_two,
#                                     y=y_test_two,
#                                     batch_size=1,
#                                     verbose=1),
#                     TAIL=y_test_two.shape[0],
#                     save_plot=True)
# %%
# Convert the model to Tensorflow Lite and save.
# with open(f'{model_loc}Model-№2-{profile}.tflite', 'wb') as f:
#     f.write(
#         tf.lite.TFLiteConverter.from_keras_model(
#                 model=gru_model
#             ).convert()
#         )
