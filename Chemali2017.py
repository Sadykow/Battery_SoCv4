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
from py_modules.tf_modules import scheduler, get_learning_rate
from py_modules.utils import str2bool, Locate_Best_Epoch
from py_modules.plotting import predicting_plot, history_plot

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

opts = [('-d', 'False'), ('-e', '100'), ('-l', '2'), ('-n', '131'), ('-a', '1'),
        ('-g', '0'), ('-p', 'FUDS')]
debug   : int = 0
batch   : int = 1
mEpoch  : int = 10
nLayers : int = 1
nNeurons: int = 262
attempt : str = '1'
GPU     : int = None
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
                        label_columns=['SoC(%)'], batch=batch,
                        includeTarget=False, normaliseLabal=False,
                        shuffleTraining=False)
ds_train, x_train, y_train = window.train
ds_valid, x_valid, y_valid = window.valid
ds_testi, x_testi, y_testi = window.test

# For training-validation if necessary 16800:24800
xt_valid = np.array(x_train[-8000:,:,:], copy=True, dtype=np.float32)
yt_valid = np.array(y_train[-8000:,:]  , copy=True, dtype=np.float32)
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

file_name : str = os.path.basename(__file__)[:-3]
model_name : str = 'Model №1'
####################! ADD model_name to path!!! ################################
model_loc : str = f'Mods/{nLayers}x{file_name}-({nNeurons})/{attempt}-{profile}/'
iEpoch = 0
firstLog : bool = True
iLr     : float = 0.001
prev_error : np.float32 = 1.0
try:
    #! Try Locating Best Epoch instead and going from it
    # for _, _, files in os.walk(model_loc):
    #     for file in files:
    #         if file.endswith('.ch'):
    #             iEpoch = int(os.path.splitext(file)[0])
    iEpoch, prev_error  = Locate_Best_Epoch(f'{model_loc}history.csv', 'mae')
    lstm_model : tf.keras.models.Sequential = tf.keras.models.load_model(
            f'{model_loc}{iEpoch}',
            compile=False)
    iLr = get_learning_rate(iEpoch, iLr)
    firstLog = False
    print(f"Model Identefied at {iEpoch} with {prev_error}. Continue training.")
except OSError as identifier:
    print("Model Not Found, initiating new. {} \n".format(identifier))
    lstm_model = create_model(
            tf.keras.layers.LSTM, layers=nLayers, neurons=nNeurons,
            dropout=0.2, input_shape=x_train.shape[-2:], batch=1
        )
    iLr = 0.001
    firstLog = True
prev_model = tf.keras.models.clone_model(lstm_model,
                                input_tensors=None, clone_function=None)

# %%
optimiser = tf.optimizers.Adam(learning_rate=iLr,
            beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
loss_fn   = tf.losses.MeanAbsoluteError(
                    reduction=tf.keras.losses.Reduction.NONE
                )
MAE     = tf.metrics.MeanAbsoluteError()
RMSE    = tf.metrics.RootMeanSquaredError()
RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)

@tf.function
def train_single_st(input : tuple[np.ndarray, np.ndarray],
                    metrics : tf.keras.metrics) -> tf.Tensor:
    # Execute model as training
    with tf.GradientTape() as tape:
        #? tf.EagerTensor['1,1', tf.float32]
        logits     : tf.Tensor = lstm_model(input[0], training=True)
        #? tf.EagerTensor['1,', tf.float32]
        loss_value : tf.Tensor = loss_fn(input[1], logits)
    
    # Get gradients and apply optimiser to model
    grads : list[tf.Tensor]= tape.gradient(
                    loss_value,
                    lstm_model.trainable_weights
                )
    optimiser.apply_gradients(zip(grads, lstm_model.trainable_weights))
    
    # Update metrics
    for metric in metrics:
        metric.update_state(y_true=input[1], y_pred=logits)
    return loss_value

@tf.function
def test_step(input : tuple[np.ndarray, np.ndarray]) -> tf.Tensor:
    return lstm_model(input, training=False)

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
        # mae[i] = val_MAE.result()
        # rmse[i] = val_RMSE.result()
        # rsquare[i] = val_RSquare.result()
    #! Error with RMSE here. No mean should be used.
    # return [loss, mae, rmse, rsquare, logits]
    return [loss, val_MAE.result(), val_RMSE.result(), val_RSquare.result(), logits]

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

n_attempts : int = 10

while iEpoch < mEpoch:
    iEpoch+=1
    # pbar = tqdm(total=y_train.shape[0])
    tic : float = time.perf_counter()
    sh_i = np.arange(y_train.shape[0])
    np.random.shuffle(sh_i)
    print(f'Commincing Epoch: {iEpoch}')
    loss_value : np.float32 = 0.0
    for i in sh_i[::]:
        loss_value = train_single_st((x_train[i,:,:,:], y_train[i,:]),
                                    [MAE,RMSE,RSquare])
        # Progress Bar
        # pbar.update(1)
        # pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
        #                         f'loss: {(loss_value[0]):.4f} - '
        #                         f'mae: {MAE.result():.4f} - '
        #                         f'rmse: {RMSE.result():.4f} - '
        #                         f'rsquare: {RSquare.result():.4f}'
        #                     )
    toc : float = time.perf_counter() - tic
    # pbar.close()
    cLr = optimiser.get_config()['learning_rate']
    print(f'Epoch {iEpoch}/{mEpoch} :: '
            f'Elapsed Time: {toc} - '
            f'loss: {loss_value[0]:.4f} - '
            f'mae: {MAE.result():.4f} - '
            f'rmse: {RMSE.result():.4f} - '
            f'rsquare: {RSquare.result():.4f} - '
            f'Lear-Rate: {cLr} - '
        )
    #! If Loss rised from previos - reload with prev_model
    #! Verefy that training is good by comparing previos error
    #? Dealing with NaN state. Give few trials to see if model improves
    curr_error = MAE.result().numpy()
    if (tf.math.is_nan(loss_value[0]) or curr_error > prev_error):
        print('->> NaN or High error model')
        i_attempts : int = 0
        firstFaltyLog : True = True
        while i_attempts < n_attempts:
            print(f'->>> Attempt {i_attempts}')
            lstm_model.save(filepath=f'{model_loc}{iEpoch}-fail-{i_attempts}',
                    overwrite=True, include_optimizer=True,
                    save_format='h5', signatures=None, options=None
            )
            lstm_model = tf.keras.models.clone_model(prev_model)
            
            np.random.shuffle(sh_i)
            # pbar = tqdm(total=y_train.shape[0])
            tic = time.perf_counter()
            for i in sh_i[::]:
                loss_value = train_single_st((x_train[i,:,:,:], y_train[i,:]),
                                        [MAE,RMSE,RSquare])
                # Progress Bar
                # pbar.update(1)
                # pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
                #                     f'loss: {(loss_value[0]):.4f} - '
                #                     )
            toc = time.perf_counter() - tic
            # pbar.close()
            #! TODO: Reduce LEarning step.
            #! TODO: withot scheduler iLr/2
            TRAIN = valid_loop((xt_valid, yt_valid), verbose = debug)
            iLr /= 2
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
            print(faulty_hist_df)
            curr_error = MAE.result().numpy()
            if (not tf.math.is_nan(loss_value[0]) and
                not curr_error > prev_error and
                not TRAIN[1] > 0.20 ):
                print(f'->>> Attempt {i_attempts} Passed')
                break
            else:
                i_attempts += 1
        if (i_attempts == n_attempts):
            print('->> Model reached the optimum -- Breaking')
            break
        else:
            print('->> Model restored -- continue training')
            lstm_model.save(filepath=f'{model_loc}{iEpoch}-{i_attempts}',
                            overwrite=True, include_optimizer=True,
                            save_format='h5', signatures=None, options=None
                )
            prev_model = tf.keras.models.clone_model(lstm_model)
            prev_error = curr_error
    else:
        lstm_model.save(filepath=f'{model_loc}{iEpoch}',
                        overwrite=True, include_optimizer=True,
                        save_format='h5', signatures=None, options=None
                )
        prev_model = tf.keras.models.clone_model(lstm_model)
        prev_error = curr_error

    # Update learning rate
    iLr = scheduler(iEpoch, iLr)
    optimiser.learning_rate = iLr

    # #! Stop using that
    # if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
    #     os.remove(f'{model_loc}{iEpoch-1}.ch')
    # os.mknod(f'{model_loc}{iEpoch}.ch')

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
        )
    # Validating model 
    val_tic : float = time.perf_counter()
    PERF = valid_loop((x_valid, y_valid), verbose = debug)
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
        )
    #! PErform testing and also save to log file
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
        save_title_type : str = 'LSTM Test on US06'
        save_file_name  : str = f'US06-{iEpoch}'
    else:
        save_title_type : str = 'LSTM Test on DST'
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
        save_title_type : str = 'LSTM Test on US06'
        save_file_name  : str = f'US06-{iEpoch}'
    else:
        save_title_type : str = 'LSTM Test on FUDS'
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
        )

    try:
        hist_df : pd.DataFrame = pd.read_csv(f'{model_loc}history.csv',
                                             index_col='Epoch')
        hist_df = hist_df.reset_index()
    except ValueError:
    ####################! TEMP FIX!!! ################################
        hist_df : pd.DataFrame = pd.read_csv(f'{model_loc}history.csv')
        hist_df.index = hist_df.index+1
        hist_df = hist_df.reset_index()
        hist_df = hist_df.rename(columns={'index' : 'Epoch'})

    #! Store the LearningRate
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
            'learn_r': np.array(iLr),
        }, name=0)
    if(len(hist_df[hist_df['Epoch']==iEpoch]) == 0):
        hist_df = hist_df.append(hist_ser, ignore_index=True)
    else:
        try:
            hist_df[hist_df['Epoch']==iEpoch] = hist_ser
        except:
            hist_df['learn_r'] = np.nan
            hist_df[hist_df['Epoch']==iEpoch] = hist_ser

    hist_df.to_csv(f'{model_loc}history.csv', index=False)
    
    # Plot History for reference and overwrite if have to    
    history_plot(profile, model_name, model_loc, hist_df, save_plot=True)

    pd.DataFrame(TRAIN[4]).to_csv(f'{model_loc}{iEpoch}-train-logits.csv')
    pd.DataFrame(PERF[4]).to_csv(f'{model_loc}{iEpoch}-valid-logits.csv')
    pd.DataFrame(np.append(TEST1[4], TEST2[4])
                ).to_csv(f'{model_loc}{iEpoch}-test--logits.csv')
# %%
#! Find the lowest error amongs all models in the file and make prediction
# history.head()
bestEpoch, _  = Locate_Best_Epoch(f'{model_loc}history.csv', 'mae')
best_model : tf.keras.models.Sequential = tf.keras.models.load_model(
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
#! Column chart for all attempts
# %%
# plt.plot(df['mae'])
# plt.plot(df['val_mae'])#['train_mae'])

# plt.plot(df['rmse'])
# plt.plot(df['val_rms'])#['train_rms'])

# %%
# def format_SoC(value, _):
#     return int(value*100)
# TAIL = y_test_one.shape[0]
# test_time = np.linspace(0, PRED.shape[0]/60, PRED.shape[0])
# skip=1
# fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
# ax1.plot(test_time[:TAIL:skip], y_test_one[::skip,-1],'-',
#         label="Actual", color='#0000ff')
# ax1.plot(test_time[:TAIL:skip],
#         PRED,'--',
#         label="Prediction", color='#ff0000')

# # ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
# ax1.set_xlabel("Time Slice (min)", fontsize=32)
# ax1.set_ylabel("SoC (%)", fontsize=32)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax2.plot(test_time[:TAIL:skip],
#         RMS,
#         label="ABS error", color='#698856')
# ax2.fill_between(test_time[:TAIL:skip],
#         RMS[:,0],
#             color='#698856')
# ax2.set_ylabel('Error', fontsize=32, color='#698856')
# ax2.tick_params(axis='y', labelcolor='#698856')
# # if profile == 'FUDS':
# #     ax1.set_title(f"{file_name} LSTM Test on US06 - {profile}-trained",
# #                 fontsize=18)
# # else:
# #     ax1.set_title(f"{file_name} LSTM Test on FUDS - {profile}-trained",
# #                 fontsize=18)
# ax1.legend(prop={'size': 32})
# ax1.tick_params(axis='both', labelsize=28)
# ax2.tick_params(axis='y', labelcolor='#698856', labelsize=28)
# ax2.legend(loc='center right', bbox_to_anchor=(1.0,0.80), prop={'size': 32})
# # ax1.set_ylim([-0.1,1.2])
# # ax2.set_ylim([-0.1,1.6])
# # fig.tight_layout()  # otherwise the right y-label is slightly clipped
# ax2.set_xlim([10,45])
# ax2.set_ylim([-0.0,1.6])
# ax1.set_title(
#     f"Accuracy visualisation example",
#     fontsize=36)
# ax1.set_ylim([0.7,1])
# ax1.set_xlim([10,50])
# # ax1.annotate('Actual SoC percent', xy=(25, 0.86),
# #             xycoords='data', fontsize=28, color='#0000ff',
# #             xytext=(0.5, 0.85), textcoords='axes fraction',
# #             arrowprops=dict(facecolor='black', shrink=0.05),
# #             horizontalalignment='right', verticalalignment='top',
# #             )
# # ax1.annotate('Predicted SoC', xy=(24, 0.77),
# #             xycoords='data', fontsize=28, color='#ff0000',
# #             xytext=(0.25, 0.25), textcoords='axes fraction',
# #             arrowprops=dict(facecolor='black', shrink=0.05),
# #             horizontalalignment='right', verticalalignment='top',
# #             )

# plt.annotate(s='', xy=(25.5,0.83), xytext=(25.5,0.43),
#                 xycoords='data', fontsize=28, 
#                 arrowprops=dict(arrowstyle='<->', facecolor='black'),
#                 horizontalalignment='right', verticalalignment='top')
# plt.annotate(s='', xy=(20.5,1.02), xytext=(20.5,0.68),
#                 xycoords='data', fontsize=28, 
#                 arrowprops=dict(arrowstyle='<->', facecolor='black'),
#                 horizontalalignment='right', verticalalignment='top')
# ax1.text(19.5, 0.855, r'$\Delta$', fontsize=24)
# ax1.text(24.5, 0.815, r'$\Delta$', fontsize=24)
# ax1.annotate('Difference area fill', xy=(26, 0.82),
#             xycoords='data', fontsize=28, color='#698856',
#             xytext=(0.95, 0.45), textcoords='axes fraction',
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             horizontalalignment='right', verticalalignment='top',
#             )
# ax1.annotate('', xy=(26, 0.72),
#             xycoords='data', fontsize=28,
#             xytext=(0.7, 0.41), textcoords='axes fraction',
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             horizontalalignment='right', verticalalignment='top',
#             )
# ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_SoC))
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
# fig.savefig(f'tests/figures/plot-example.svg')
# # Cleaning Memory from plots
# fig.clf()
# plt.close()
# %%
# https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras
#! Use pruning to get rid of extra things and make smaller model
# Convert the model to Tensorflow Lite and save.
# with open(f'{model_loc}Model-№1-{profile}.tflite', 'wb') as f:
#     f.write(
#         tf.lite.TFLiteConverter.from_keras_model(
#                 model=lstm_model
#             ).convert()
    #         )