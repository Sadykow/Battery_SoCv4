#!/usr/bin/python
# %% [markdown]
# # # 1
# # #
# # GRU for SoC by Meng Jiao 2020 based momentum optimized algorithm

# In the mmentum gradient method, the current weight change direction takes a
#compromise of the gradient direction at current instant and at historical time
#to prevent the oscilation of weight change and to improve the SoC estimation
#speed.

# Random noise are added to the sampled data, o as to prevent overfitting of the
#GRU model.

# The amprere-hour integration method computes the SoC by integrating the
#current over time [6,7]. It can be easily realized, but this open-loop SoC
#estimation may deviate from the true SoC due to inaccurate SiC initialization.
# The momentum gradient algorithm is proposed to solve oscilation problem
#occured in weight updating if the GRU model during the grafient algorithm
#implementing.

# Takes following vector as input:
# Ψ k = [V (k), I(k)], where V (k), I(k)
# And outputs:
# output SoC as 

# The classic Gradient algorithm in Tensorflow refered as SGD.
# The momentum gradient algorithm takes into account the gradient at current
#instant and at historical time.

# Data Wnt through MinMax normalization. All of them. To eliminate the influence
#of the dimensionality between data. Reduce the network load and improve speed.
# Solve overfitting by adding Random noise??? Chego blin?

# Structure of model = (Input(1,2)-> GRU(units, Stateful) -> Output(1))
# Neuron number - Lowest RMSE 30.

# The cost funtion:
# E = 0.5*(y_true-y_pred)^2     (13)
# Optimizer SGD - Singma of momentum of 0.03-0.10 gives best results it seems.
#0.20 makes overshoot.
# Standard metricks of MAE and RMSE. Additional Custom metric of R2

# R2 = 1 - (sum((y_true-y_pred)^2))/((sum(y_true-y_avg))^2)

# 400 Epochs until RMSE = 0.150 or MAE = 0.0076
# %%
import datetime
import logging
import os, sys, getopt    # OS, SYS, argc functions
from sys import platform  # Get type of OS

import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
plt.switch_backend('agg')       #! FIX in the no-X env: RuntimeError: Invalid DISPLAY variable
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

import gc           # Garbage Collector
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

opts = [('-d', 'False'), ('-e', '100'), ('-l', '3'), ('-n', '131'), ('-a', '11'),
        ('-g', '1'), ('-p', 'FUDS')] # 2x131 1x1572 
debug   : int = 0
batch   : int = 1
mEpoch  : int = 10
nLayers : int = 1
nNeurons: int = 262
attempt : str = '1'
GPU     : int = None
profile : str = 'DST'
rounding: int = 5
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

    if GPU == 1:
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
# Getting Data from excel files.
# float_dtype : type = np.float32
# train_dir : str = 'Data/A123_Matt_Set'
# valid_dir : str = 'Data/A123_Matt_Val'
# columns   : list[str] = [
#                         'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
#                         'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
#                     ]

# def Read_Excel_File(path : str, profile : range,
#                     columns : list[str]) -> pd.DataFrame:
#     """ Reads Excel File with all parameters. Sheet Name universal, columns,
#     type taken from global variables initialization.

#     Args:
#         path (str): Path to files with os.walk

#     Returns:
#         pd.DataFrame: Single File frame.
#     """
#     try:
#       df : pd.DataFrame = pd.read_excel(io=path,
#                         sheet_name='Channel_1-006',
#                         header=0, names=None, index_col=None,
#                         usecols=['Step_Index'] + columns,
#                         squeeze=False,
#                         dtype=float_dtype,
#                         engine='openpyxl', converters=None, true_values=None,
#                         false_values=None, skiprows=None, nrows=None,
#                         na_values=None, keep_default_na=True, na_filter=True,
#                         verbose=False, parse_dates=False, date_parser=None,
#                         thousands=None, comment=None, skipfooter=0,
#                         convert_float=True, mangle_dupe_cols=True
#                       )
#     except:
#       df : pd.DataFrame = pd.read_excel(io=path,
#                         sheet_name='Channel_1-005',
#                         header=0, names=None, index_col=None,
#                         usecols=['Step_Index'] + columns,
#                         squeeze=False,
#                         dtype=float_dtype,
#                         engine='openpyxl', converters=None, true_values=None,
#                         false_values=None, skiprows=None, nrows=None,
#                         na_values=None, keep_default_na=True, na_filter=True,
#                         verbose=False, parse_dates=False, date_parser=None,
#                         thousands=None, comment=None, skipfooter=0,
#                         convert_float=True, mangle_dupe_cols=True
#                       )
#     df = df[df['Step_Index'].isin(profile)]
#     df = df.reset_index(drop=True)
#     df = df.drop(columns=['Step_Index'])
#     df = df[columns]   # Order columns in the proper sequence
#     return df

# def diffSoC(chargeData   : pd.Series,
#             discargeData : pd.Series) -> pd.Series:
#     """ Return SoC based on differnece of Charge and Discharge Data.
#     Data in range of 0 to 1.
#     Args:
#         chargeData (pd.Series): Charge Data Series
#         discargeData (pd.Series): Discharge Data Series

#     Raises:
#         ValueError: If any of data has negative
#         ValueError: If the data trend is negative. (end-beg)<0.

#     Returns:
#         pd.Series: Ceil data with 2 decimal places only.
#     """
#     # Raise error
#     if((any(chargeData) < 0)
#         |(any(discargeData) < 0)):
#         raise ValueError("Parser: Charge/Discharge data contains negative.")
#     return np.round((chargeData - discargeData)*100)/100

# #? Getting training data and separated file by batch
# for _, _, files in os.walk(train_dir):
#     files.sort(key=lambda f: int(f[-13:-5])) # Sort by last dates
#     # Initialize empty structures
#     train_X : list[pd.DataFrame] = []
#     train_Y : list[pd.DataFrame] = []
#     for file in files[:]:
#         X : pd.DataFrame = Read_Excel_File(train_dir + '/' + file,
#                                     range(22,25), columns) #! or 21
#         Y : pd.DataFrame = pd.DataFrame(
#                 data={'SoC' : diffSoC(
#                             chargeData=X.loc[:,'Charge_Capacity(Ah)'],
#                             discargeData=X.loc[:,'Discharge_Capacity(Ah)']
#                             )},
#                 dtype=float_dtype
#             )
#         X = X[['Current(A)', 'Voltage(V)']]
#         train_X.append(X)
#         train_Y.append(Y)
# # %%
# look_back : int = 1
# scaler_MM : MinMaxScaler    = MinMaxScaler(feature_range=(0, 1))
# scaler_CC : MinMaxScaler    = MinMaxScaler(feature_range=(0, 1))
# scaler_VV : StandardScaler  = StandardScaler()
# def roundup(x : float, factor : int = 10) -> int:
#     """ Round up to a factor. Uses it to create hidden neurons, or Buffer size.
#     TODO: Make it a smarter rounder.
#     Args:
#         x (float): Original float value.
#         factor (float): Factor towards which it has to be rounder

#     Returns:
#         int: Rounded up value based on factor.
#     """
#     if(factor == 10):
#         return int(np.ceil(x / 10)) * 10
#     elif(factor == 100):
#         return int(np.ceil(x / 100)) * 100
#     elif(factor == 1000):
#         return int(np.ceil(x / 1000)) * 1000
#     else:
#         print("Factor of {} not implemented.".format(factor))
#         return None

# def create_Batch_dataset(X : list[np.ndarray], Y : list[np.ndarray],
#                     look_back : int = 1
#                     ) -> tuple[np.ndarray, np.ndarray]:
    
#     batch : int = len(X)
#     dataX : list[np.ndarray] = []
#     dataY : list[np.ndarray] = []
    
#     for i in range(0, batch):
#         d_len : int = X[i].shape[0]-look_back
#         dataX.append(np.zeros(shape=(d_len, look_back, X[i].shape[1]),
#                     dtype=float_dtype))
#         dataY.append(np.zeros(shape=(d_len,), dtype=float_dtype))    
#         for j in range(0, d_len):
#             #dataX[i, j, :, :] = dataset[i:(i+look_back), j:j+1]
#             #dataY[i, j]       = dataset[i + look_back, j:j+1]
#             dataX[i][j,:,:] = X[i][j:(j+look_back), :]  
#             dataY[i][j]     = Y[i][j+look_back,]
#     return dataX, dataY

# sample_size : int = 0
# for i in range(0, len(train_X)):
#     #! Scale better with STD on voltage
#     #train_X[i].iloc[:,0] = scaler_CC.fit_transform(np.expand_dims(train_X[i]['Current(A)'], axis=1))
#     #train_X[i].iloc[:,1] = scaler_VV.fit_transform(np.expand_dims(train_X[i]['Voltage(V)'], axis=1))    
#     train_Y[i] = scaler_MM.fit_transform(train_Y[i])
#     train_X[i] = train_X[i].to_numpy()
#     sample_size += train_X[i].shape[0]
    

# trX, trY = create_Batch_dataset(train_X, train_Y, look_back)
# %%
Data : str = ''
if(platform=='win32'):
    Data = 'DataWin\\'
else:
    Data = 'Data/'
#TODO -------------------- FIX THE SET BEFORE LEAVING!!!!
dataGenerator = DataGenerator(train_dir=f'{Data}A123_Matt_Single',
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
#! Test with golbal
def create_model(mFunc : Callable, layers : int = 1,
                 neurons : int = 500, dropout : float = 0.2,
                 input_shape : tuple = (500, 3), batch : int = 1
            ) -> tf.keras.models.Sequential:
    """ Creates Tensorflow 2 based time series models with inputs exception 
    handeling. Accepts multilayer models.
    TODO: For Py3.10 correct typing with: func( .. dropout : int | float .. )

    Args:
        mFunc (Callable): Time series model function. .GRU or .GRU
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
model_name : str = 'ModelsUp-5'
####################! ADD model_name to path!!! ################################
model_loc : str = f'Modds/{model_name}/{nLayers}x{file_name}-({nNeurons})/{attempt}-{profile}/'
iEpoch = 0
firstLog : bool = True
iLr     : float = 0.001
prev_error : np.float32 = 1.0
try:
    iEpoch, prev_error  = Locate_Best_Epoch(f'{model_loc}history.csv', 'mae')
    model : tf.keras.models.Sequential = tf.keras.models.load_model(
            f'{model_loc}{iEpoch}',
            compile=False)
    iLr = get_learning_rate(iEpoch, iLr, 'linear')
    firstLog = False
    print(f"Model Identefied at {iEpoch} with {prev_error}. Continue training.")
except (OSError, TypeError) as identifier:
    print("Model Not Found, initiating new. {} \n".format(identifier))
    if type(x_train) == list:
        input_shape : tuple = x_train[0].shape[-2:]
    else:
        input_shape : tuple = x_train.shape[-2:]
    model = create_model(
            tf.keras.layers.GRU, layers=nLayers, neurons=nNeurons,
            dropout=0.2, input_shape=input_shape, batch=1
        )
    iLr = 0.001
    firstLog = True
prev_model = tf.keras.models.clone_model(model) 

# %%
optimiser = tf.keras.optimizers.SGD(
                learning_rate=0.001, momentum=0.3,
            nesterov=False, name='SGDwM')
loss_fn   = tf.losses.MeanAbsoluteError(
                    reduction=tf.keras.losses.Reduction.NONE,
                    #from_logits=True
                )
MAE     = tf.metrics.MeanAbsoluteError()
RMSE    = tf.metrics.RootMeanSquaredError()
RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)
# CR_ER   = tf.metrics.MeanAbsoluteError()

@tf.function
def train_single_st(input : tuple[np.ndarray, np.ndarray],
                    metrics : tf.keras.metrics,
                    # curr_error : tf.keras.metrics
                    ) -> tf.Tensor:
    # Execute model as training
    with tf.GradientTape() as tape:
        #? tf.EagerTensor['1,1', tf.float32]
        logits     : tf.Tensor = model(input[0], training=True)
        #? tf.EagerTensor['1,', tf.float32]
        loss_value : tf.Tensor = loss_fn(input[1], logits)
    
    # Get gradients and apply optimiser to model
    grads : list[tf.Tensor] = tape.gradient(
                    loss_value,
                    model.trainable_weights
                )
    optimiser.apply_gradients(zip(grads, model.trainable_weights))
    
    # Update metrics before
    for metric in metrics:
        metric.update_state(y_true=input[1], y_pred=logits)

    return loss_value

@tf.function
def test_step(input : tuple[np.ndarray, np.ndarray]) -> tf.Tensor:
    return model(input, training=False)

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
    mae      : float = val_MAE.result()
    rmse     : float = val_RMSE.result()
    r_Square : float = val_RSquare.result()
    
    # Reset training metrics at the end of each epoch
    val_MAE.reset_states()
    val_RMSE.reset_states()
    val_RSquare.reset_states()

    return [loss, mae, rmse, r_Square, logits]

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

n_attempts : int = 20
while iEpoch < mEpoch:
    iEpoch+=1
#! Fit one sample - PRedict next one like model() - read all through file output
#!and hope this makes a good simulation of how online learning will go.
    pbar = tqdm(total=y_train.shape[0])
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
        pbar.update(1)
        pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
                                # f'loss: {(loss_value[0]):.4f} - '
                                f'mae: {MAE.result():.4f} - '
                                f'rmse: {RMSE.result():.4f} - '
                                # f'rsquare: {RSquare.result():.4f} --- '
                            )
    toc : float = time.perf_counter() - tic
    pbar.close()
    cLr = optimiser.get_config()['learning_rate']
    print(f'Epoch {iEpoch}/{mEpoch} :: '
            f'Elapsed Time: {toc} - '
            # f'loss: {loss_value[0]:.4f} - '
            f'mae: {MAE.result():.4f} - '
            f'rmse: {RMSE.result():.4f} - '
            f'rsquare: {RSquare.result():.4f} - '
            f'Lear-Rate: {cLr} - '
        )
#! In Mamo methods implement Callback to reset model after 500 steps and then 
#!step by one sample for next epoch to capture shift in data. Hell method, but
#!might be more effective that batching 12 together.
    #* Dealing with NaN state. Give few trials to see if model improves
    curr_error = MAE.result().numpy()
    # curr_error = CR_ER.result().numpy()
    print(f'The post optimiser error: {curr_error}', flush=True)
    if (tf.math.is_nan(loss_value[0]) or curr_error > prev_error):
        print('->> NaN or High error model')
        i_attempts : int = 0
        firstFaltyLog : bool = True
        while i_attempts < n_attempts:
            print(f'->>> Attempt {i_attempts}')
            try:
                model.save(filepath=f'{model_loc}{iEpoch}-fail-{i_attempts}',
                        overwrite=True, include_optimizer=True,
                        save_format='h5', signatures=None, options=None
                )
            except OSError:
                os.remove(f'{model_loc}{iEpoch}-fail-{i_attempts}')
                model.save(filepath=f'{model_loc}{iEpoch}-fail-{i_attempts}',
                        overwrite=True, include_optimizer=True,
                        save_format='h5', signatures=None, options=None
                )
            model = tf.keras.models.clone_model(prev_model)
            
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
            iLr /= 2
            optimiser.learning_rate = iLr
            print(f">>> Updating iLR with {iLr}")

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
        model.save(filepath=f'{model_loc}{iEpoch}',
                       overwrite=True, include_optimizer=True,
                       save_format='h5', signatures=None, options=None
                )
        prev_model = tf.keras.models.clone_model(model)
        prev_error = curr_error

    # Update learning rate
    iLr = scheduler(iEpoch, iLr, 'linear')
    optimiser.learning_rate = iLr
    
    # Validating trained model 
    TRAIN = valid_loop((xt_valid, yt_valid), verbose = debug)
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                yt_valid[:,0,0]-TRAIN[4])))
    predicting_plot(profile=profile, file_name=model_name,
                    model_loc=f'{model_loc}/traiPlots/',
                    model_type='GRU valid',
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
                    model_type='GRU valid',
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

    # Collect garbage leftovers
    gc.collect()
# %%
# tr_pred = np.zeros(shape=(7094,))
# i = 0
# with open("pp-temp.txt", "r") as file1:
#     for line in file1.readlines():
#         tr_pred[i] = float(line.split('[[')[1].split(']]\n')[0])
#         i += 1

# plt.plot(tr_pred)
# plt.plot(trY[0])
# %%
# def smooth(y, box_pts: int) -> np.array:
#     """ Smoothing data using numpy convolve. Based on the size of the
#     averaging box, data gets smoothed.
#     Here it used in following form:
#     y = V/(maxV-minV)
#     box_pts = 500

#     Args:
#         y (pd.Series): A data which requires to be soothed.
#         box_pts (int): Number of points to move averaging box

#     Returns:
#         np.array: Smoothed data array
#     """
#     box = np.ones(box_pts)/box_pts
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth
# # 5 seconds timestep
# plt.figure()
# plt.plot(train_X[0][:,1])
# plt.plot(smooth(train_X[0][:,1], 150))
# #plt.xlim([0, 7000])
# plt.ylim([2.5, 3.7])
# plt.xlabel('TimeSteps (s)')
# plt.ylabel('Voltage (V)')
# plt.grid()
# plt.show()
# %%
# Plot
# import seaborn as sns
# train_X[0]['Time (s)'] = np.linspace(0,7095*5,7095)
# g = sns.relplot(x='Time (s)', y='Temperature (C)_1', kind="line",
#                 data=train_X[0], size=11, color='k')
# #plt.xlim(-100, 40000)
# #plt.ylim(2.25, 3.75)
# g.fig.autofmt_xdate()
# fir = g.fig
# fir.savefig('../1-Voltage.svg', transparent=True)
# # tr_pred = np.zeros(shape=(7094,))
# i = 0
# with open("tt-temp.txt", "r") as file1:
#     for line in file1.readlines():
#         tr_pred[i] = float(line.split('[[')[1].split(']]\n')[0])
#         i += 1
# plt.plot(tr_pred)
# plt.plot(trY[0])
# # %%
# epochs : int = 6 #! 37*12 seconds = 444s
# file_path = 'Models/Stateful/GRU_test11_SOC'
# for i in range(1,epochs+1):
#     print(f'Epoch {i}/{epochs}')
#     for i in range(0, len(trX)):
#         history = model.fit(trX[i], trY[i], epochs=1, batch_size=1,
#                 verbose=1, shuffle=False)
#         #model.reset_states()    #! Try next time without reset
#     # for j in range(0,trX.shape[0]):
#     #     model.train_on_batch(trX[j,:,:,:], trY[j,:])
#     #! Wont work. Needs a Callback for that.
#     # if(i % train_df.shape[0] == 0):
#     #     print("Reseting model")
#     if(history.history['root_mean_squared_error'][0] < min_rmse):
#         min_rmse = history.history['root_mean_squared_error'][0]
#         model.save(file_path)

#     #histories.append(history)
    
    
#     # Saving history variable
#     # convert the history.history dict to a pandas DataFrame:     
#     hist_df = pd.DataFrame(history.history)
#     # # or save to csv:
#     with open('Models/Stateful/GRU_test11_SOC-history.csv', mode='a') as f:
#         if(firtstEpoch):
#             hist_df.to_csv(f, index=False)
#             firtstEpoch = False
#         else:
#             hist_df.to_csv(f, index=False, header=False)
# model.save('Models/Stateful/GRU_test11_SOC_Last')
# %%
# Convert the model to Tensorflow Lite and save.
# with open(f'{model_loc}Model-№5-{profile}.tflite', 'wb') as f:
#     f.write(
#         tf.lite.TFLiteConverter.from_keras_model(
#                 model=model
#             ).convert()
#         )