#!/usr/bin/python
# %% [markdown]
# # Auto-regression implementation (Forward-Teaching)
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
from py_modules.AutoFeedBack import AutoFeedBack
from py_modules.RobustAdam import RobustAdam
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
#     opts, args = getopt.getopt(sys.argv[1:],"hd:e:g:p:s:",
#                     ["help", "debug=", "epochs=",
#                      "gpu=", "profile="])
# except getopt.error as err: 
#     # output error, and return with an error code 
#     print (str(err)) 
#     print ('EXEPTION: Arguments requied!')
#     sys.exit(2)

opts = [('-d', 'False'), ('-e', '100'), ('-l', '2'), ('-n', '524'), ('-a', '4'),
        ('-g', '0'), ('-p', 'FUDS'), ('-s', '30')] # *x524
debug   : int = 0
batch   : int = 1
mEpoch    : int = 10
nLayers : int = 1
nNeurons: int = 262
attempt : str = '1'
GPU       : int = 0
profile   : str = 'DST'
out_steps : int = 10
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
    elif opt in ("-s", "--steps"):
        out_steps = int(arg)
# %%
# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
# mpl.rcParams['font.family'] = 'Bender'

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
Data : str = ''
if(platform=='win32'):
    Data = 'DataWin\\'
else:
    Data = 'Data/'
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
                        input_width=500, label_width=out_steps, shift=1,
                        input_columns=['Current(A)', 'Voltage(V)',
                                                'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,
                        includeTarget=True, normaliseLabal=False,
                        shuffleTraining=False)
x_train, y_train = window.train
x_valid, y_valid = window.valid
x_testi, y_testi = window.test

tv_length = len(x_valid)
xt_valid = np.array(x_train[-tv_length:,:,:], copy=True, dtype=np.float32)
yt_valid = np.array(y_train[-tv_length:,:]  , copy=True, dtype=np.float32)
# %%
file_name : str = os.path.basename(__file__)[:-3]
model_name : str = 'Novels-№2'
model_loc : str = f'Modds/{model_name}/{nLayers}x{file_name}-({nNeurons})/{attempt}-{profile}/'
iEpoch = 0
firstLog  : bool = True
iLr     : float = 0.001
prev_error : np.float32 = 1.0
try:
    iEpoch, prev_error  = Locate_Best_Epoch(f'{model_loc}history.csv', 'mae')
    lstm_model : AutoFeedBack = AutoFeedBack(units=nNeurons,
            out_steps=out_steps, num_features=1
        )
    lstm_model.load_weights(f'{model_loc}{iEpoch}/{iEpoch}')
    iLr = get_learning_rate(iEpoch, iLr, 'linear')
    firstLog = False
    prev_model : AutoFeedBack = AutoFeedBack(units=nNeurons,
                out_steps=out_steps, num_features=1
            )
    prev_model.load_weights(f'{model_loc}{iEpoch}/{iEpoch}')
    print("Model Identefied. Continue training.")
except:
    print("Model Not Found, with some TF error.\n")
    lstm_model : AutoFeedBack = AutoFeedBack(units=nNeurons,
            out_steps=out_steps, num_features=1
        )
    # lstm_model : AutoFeedBack = tf.keras.models.load_model(
    #         f'{model_loc}{iEpoch}-tf',
    #         compile=False,
    #         custom_objects={"RSquare": tfa.metrics.RSquare,
    #                         "AutoFeedBack": AutoFeedBack
    #                         })
    iLr = 0.001
    firstLog = True
    prev_model : AutoFeedBack = AutoFeedBack(units=nNeurons,
                out_steps=out_steps, num_features=1
            )
# %%
# optimiser = RobustAdam(learning_rate = iLr)
optimiser = tf.optimizers.Adam(learning_rate=iLr,
            beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
loss_fn   = tf.losses.MeanAbsoluteError(
                    reduction=tf.keras.losses.Reduction.NONE,
                    #from_logits=True
                )
MAE     = tf.metrics.MeanAbsoluteError()
RMSE    = tf.metrics.RootMeanSquaredError()
RSquare = tfa.metrics.RSquare(y_shape=(out_steps,), dtype=tf.float32)
# CR_ER   = tf.metrics.MeanAbsoluteError()

@tf.function
def train_single_st(input : tuple[np.ndarray, np.ndarray],
                    metrics : tf.keras.metrics,
                    # curr_error : tf.keras.metrics
                    ) -> tf.Tensor:
    # Execute model as training
    with tf.GradientTape() as tape:
        #? tf.EagerTensor['1,1', tf.float32]
        logits     : tf.Tensor = lstm_model(input[0], training=True)
        #? tf.EagerTensor['1,', tf.float32]
        loss_value : tf.Tensor = loss_fn(input[1], logits)
    
    # Get gradients and apply optimiser to model
    grads : list[tf.Tensor] = tape.gradient(
                    loss_value,
                    lstm_model.trainable_weights
                )
    # optimiser.update_loss(prev_loss=curr_error, current_loss=loss_value)
    optimiser.apply_gradients(zip(grads, lstm_model.trainable_weights))
    
    # Update metrics before
    for metric in metrics:
        metric.update_state(y_true=input[1], y_pred=logits)

    # Currrent error tracker
    # curr_error.update_state(y_true=input[1],
    #                         y_pred=lstm_model(input[0], training=False))
    
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
    # val_RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)
    
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
        # val_RSquare.update_state(y_true=y[i], y_pred=logits[i])
        
        loss[i] = loss_fn(y[i], logits[i])

    # Error with RMSE here. No mean should be used.
    mae      : float = val_MAE.result()
    rmse     : float = val_RMSE.result()
    # r_Square : float = val_RSquare.result()
    
    # Reset training metrics at the end of each epoch
    val_MAE.reset_states()
    val_RMSE.reset_states()
    # val_RSquare.reset_states()

    return [loss, mae, rmse, 0, logits]

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
def format_func(value, _):
    return int(value*100)

def forward_plot(profile, out_steps, y_valid, model_loc, iEpoch, MAE, RMS, PRED):
    # Time range
    test_time = np.linspace(0, PRED.shape[0]/60, PRED.shape[0])


    fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
    ax1.plot(test_time, y_valid[::,0,-1], '-', label="Actual")
    ax1.plot(test_time, PRED, '--', label="Prediction")
    # ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
    ax1.set_xlabel("Time Slice (min)", fontsize=32)
    ax1.set_ylabel("SoC (%)", fontsize=32)
    #if RMS_plot:
    ax2 = ax1.twinx()
    ax2.plot(test_time,
        RMS,
        label="RMS error", color='#698856')
    ax2.fill_between(test_time,
        RMS,
        color='#698856')
    ax2.set_ylabel('Error', fontsize=32, color='#698856')
    ax2.tick_params(axis='y', labelcolor='#698856', labelsize=28)
    ax2.set_ylim([-0.1,1.6])
    ax1.set_title(
        #f"{file_name} {model_type}. {profile}-trained",
        #f"4-feature Model №2 Train. {profile}-trained. {out_steps}-steps",
        f"4-feature Model №2 LSTM Feed-forward {profile}-tr,  {out_steps}-steps",
        fontsize=36)
    ax1.legend(prop={'size': 32})
    ax2.legend(loc='center right', bbox_to_anchor=(1.0,0.80), prop={'size': 32})
    ax1.tick_params(axis='both', labelsize=28)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax1.set_ylim([-0.1,1.2])
    fig.tight_layout()
    textstr = '\n'.join((
        '$MAE  = {0:.2f}%$'.format(np.mean(MAE)*100, ),
        '$RMSE = {0:.2f}%$'.format(np.mean(RMS)*100, )
        # '$R2  = nn.nn%$'
            ))
    ax1.text(0.66, 0.74, textstr, transform=ax1.transAxes, fontsize=30,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.savefig(f'{model_loc}{profile}-FF-{iEpoch}.svg')
    fig.clf()
    plt.close()
# %%
if not os.path.exists(f'{model_loc}'):
    os.makedirs(f'{model_loc}')
if not os.path.exists(f'{model_loc}traiPlots'):
    os.mkdir(f'{model_loc}traiPlots')
if not os.path.exists(f'{model_loc}traiFfPlots'):
    os.mkdir(f'{model_loc}traiFfPlots')
if not os.path.exists(f'{model_loc}valdPlots'):
    os.mkdir(f'{model_loc}valdPlots')
if not os.path.exists(f'{model_loc}valdFfPlots'):
    os.mkdir(f'{model_loc}valdFfPlots')
if not os.path.exists(f'{model_loc}testPlots'):
    os.mkdir(f'{model_loc}testPlots')
if not os.path.exists(f'{model_loc}testFfPlots'):
    os.mkdir(f'{model_loc}testFfPlots')
if not os.path.exists(f'{model_loc}history.csv'):
    print("History not created. Making")
    with open(f'{model_loc}history.csv', mode='w') as f:
        f.write('Epoch,loss,mae,rmse,rsquare,time(s),'
                'train_l,train_mae,train_rms,train_r_s,'
                'vall_l,val_mae,val_rms,val_r_s,val_t_s,'
                'test_l,tes_mae,tes_rms,tes_r_s,tes_t_s,learn_r,'
                'FF_tr,FF_vl,FF_ts\n')
if not os.path.exists(f'{model_loc}history-cycles.csv'):
    print("Cycle-History not created. Making")
    with open(f'{model_loc}history-cycles.csv', mode='w') as f:
        f.write('Epoch,Cycle,'
                'train_l,train_mae,train_rms,train_t_s,'
                'vall_l,val_mae,val_rms,val_t_s,'
                'learn_r,FF_tr,FF_vl,FF_ts\n')
if not os.path.exists(f'{model_loc}cycles-log'):
    os.mkdir(f'{model_loc}cycles-log')
n_attempts : int = 10

while iEpoch < mEpoch:
    iEpoch+=1
    # pbar = tqdm(total=y_train.shape[0])
    tic : float = time.perf_counter()
    sh_i = np.arange(y_train.shape[0])
    np.random.shuffle(sh_i)
    print(f'Commincing Epoch: {iEpoch}')
    loss_value : np.float32 = 0.0
    # optimiser.update_loss(prev_loss=prev_error, current_loss=loss_value)
    for i in sh_i[::]:
        loss_value = train_single_st((x_train[i,:,:,:], y_train[i,:,:]),
                                    metrics=[MAE,RMSE], #RSquare
                                    # curr_error=loss_value
                                    )
        # Progress Bar
        # pbar.update(1)
        # pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
        #                         # f'loss: {(loss_value):.4f} - '
        #                         f'mae: {MAE.result():.4f} - '
        #                         f'rmse: {RMSE.result():.4f} - '
        #                         # f'rsquare: {RSquare.result():.4f} --- '
        #                     )
    toc : float = time.perf_counter() - tic
    # pbar.close()
    cLr = optimiser.get_config()['learning_rate']
    print(f'Epoch {iEpoch}/{mEpoch} :: '
            f'Elapsed Time: {toc} - '
            f'loss: {loss_value[0]:.4f} - '
            f'mae: {MAE.result():.4f} - '
            f'rmse: {RMSE.result():.4f} - '
            # f'rsquare: {RSquare.result():.4f} - '
            f'Lear-Rate: {cLr} - '
        )
    #* Dealing with NaN state. Give few trials to see if model improves
    curr_error = MAE.result().numpy()
    # curr_error = CR_ER.result().numpy()
    print(f'The post optimiser error: {curr_error}', flush=True)

    #! Finish up Nan case
    if (tf.math.is_nan(loss_value[0]) or curr_error > prev_error):
        pass
    else:
        lstm_model.save_weights(filepath=f'{model_loc}{iEpoch}/{iEpoch}',
            overwrite=True, save_format='tf', options=None
        )
        prev_model.load_weights(f'{model_loc}{iEpoch}/{iEpoch}')
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
                    model_type='LSTM valid',
                    iEpoch=f'tra-{iEpoch}',
                    Y=yt_valid[:,0, 0],
                    PRED=TRAIN[4],
                    RMS=RMS,
                    val_perf=[np.mean(TRAIN[0]), TRAIN[1],
                            TRAIN[2], TRAIN[3]],
                    TAIL=yt_valid.shape[0],
                    save_plot=True)
    print(f'Epoch {iEpoch}/{mEpoch} :: TRAIN :: '
            f'mae: {TRAIN[1]:.4f} - '
            f'rmse: {TRAIN[2]:.4f} - '
            # f'rsquare: {TRAIN[3]:.4f} - '
            f'\n'
        )
    #* Validating trained model with Feed-Forward
    VIT_input = xt_valid[0,:,:,:3]
    SOC_input = xt_valid[0,:,:,3:]
    PRED = np.zeros(shape=(yt_valid.shape[0],), dtype=np.float32)
    for i in range(1, yt_valid.shape[0]):
        logits = test_step(input=np.concatenate(
                            (VIT_input, SOC_input),
                            axis=2)
            )
        VIT_input = xt_valid[i,:,:,:3]
        SOC_input[:,:-1,:] = SOC_input[:,1:,:]
        SOC_input[:,-1,:] = logits
        PRED[i-1] = logits
    MAE_FF_tr = np.mean(tf.keras.backend.abs(yt_valid[::,0,-1]-PRED))
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                yt_valid[::,0,-1]-PRED)))
    forward_plot(profile, out_steps, yt_valid, f'{model_loc}traiFfPlots/',
                 iEpoch, MAE_FF_tr, RMS, PRED)

    # Validating model 
    val_tic : float = time.perf_counter()
    PERF = valid_loop((x_valid, y_valid), verbose = debug)
    val_toc : float = time.perf_counter() - val_tic
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_valid[:,0,0]-PERF[4])))
    predicting_plot(profile=profile, file_name=model_name,
                    model_loc=f'{model_loc}valdPlots/',
                    model_type='LSTM valid',
                    iEpoch=f'val-{iEpoch}',
                    Y=y_valid[:,0,0],
                    PRED=PERF[4],
                    RMS=RMS,
                    val_perf=[np.mean(PERF[0]), PERF[1],
                            PERF[2], PERF[3]],
                    TAIL=y_valid.shape[0],
                    save_plot=True)
    #* Validating trained model with Feed-Forward
    VIT_input = x_valid[0,:,:,:3]
    SOC_input = x_valid[0,:,:,3:]
    PRED = np.zeros(shape=(y_valid.shape[0],), dtype=np.float32)
    for i in range(1, y_valid.shape[0]):
        logits = test_step(input=np.concatenate(
                            (VIT_input, SOC_input),
                            axis=2)
            )
        VIT_input = x_valid[i,:,:,:3]
        SOC_input[:,:-1,:] = SOC_input[:,1:,:]
        SOC_input[:,-1,:] = logits
        PRED[i-1] = logits
    MAE_FF_vl = np.mean(tf.keras.backend.abs(y_valid[::,0,-1]-PRED))
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_valid[::,0,-1]-PRED)))
    forward_plot(profile, out_steps, y_valid, f'{model_loc}valdFfPlots/',
                 iEpoch, MAE_FF_vl, RMS, PRED)

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
    VIT_input = x_testi[0,:,:,:3]
    SOC_input = x_testi[0,:,:,3:]
    PRED = np.zeros(shape=(y_testi[:mid_one].shape[0],), dtype=np.float32)
    for i in range(1, y_testi[:mid_one].shape[0]):
        logits = test_step(input=np.concatenate(
                            (VIT_input, SOC_input),
                            axis=2)
            )
        VIT_input = x_testi[i,:,:,:3]
        SOC_input[:,:-1,:] = SOC_input[:,1:,:]
        SOC_input[:,-1,:] = logits
        PRED[i-1] = logits
    MAE_FF_ts1 = np.mean(tf.keras.backend.abs(y_testi[:mid_one,0,-1]-PRED))
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_testi[:mid_one,0,-1]-PRED)))
    forward_plot(profile, out_steps, y_testi[:mid_one],
                 f'{model_loc}testFfPlots/',
                 save_file_name, MAE_FF_ts1, RMS, PRED)

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
            # f'rsquare: {np.mean(np.append(TEST1[3], TEST2[3])):.4f} - '
            f'\n'
        )
    VIT_input = x_testi[mid_two,:,:,:3]
    SOC_input = x_testi[mid_two,:,:,3:]
    PRED = np.zeros(shape=(y_testi[mid_two:].shape[0],), dtype=np.float32)
    for i in range(1, y_testi[mid_two:].shape[0]):
        logits = test_step(input=np.concatenate(
                            (VIT_input, SOC_input),
                            axis=2)
            )
        VIT_input = x_testi[mid_two+i,:,:,:3]
        SOC_input[:,:-1,:] = SOC_input[:,1:,:]
        SOC_input[:,-1,:] = logits
        PRED[i-1] = logits
    MAE_FF_ts2 = np.mean(tf.keras.backend.abs(y_testi[mid_two:,0,-1]-PRED))
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_testi[mid_two:,0,-1]-PRED)))
    forward_plot(profile, out_steps, y_testi[mid_two:],
                 f'{model_loc}testFfPlots/',
                 save_file_name, MAE_FF_ts2, RMS, PRED)

    hist_df : pd.DataFrame = pd.read_csv(f'{model_loc}history.csv',
                                            index_col='Epoch')
    hist_df = hist_df.reset_index()
    #! Add more columns with FF perdormance. Find something!
    hist_ser = pd.Series(data={
            'Epoch'  : iEpoch,
            'loss'   : np.array(loss_value[0]),
            'mae'    : np.array(MAE.result()),
            'rmse'   : np.array(RMSE.result()),
            'rsquare': 0,
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
            'FF_tr'  : MAE_FF_tr,
            'FF_vl'  : MAE_FF_vl,
            'FF_ts'  : np.mean([MAE_FF_ts1, MAE_FF_ts2])
        }, name=0)
    if(len(hist_df[hist_df['Epoch']==iEpoch]) == 0):
        hist_df = hist_df.append(hist_ser, ignore_index=True)
    else:
        hist_df.loc[iEpoch-1, :] = hist_ser

    hist_df.to_csv(f'{model_loc}history.csv', index=False)
    
    # Plot History for reference and overwrite if have to    
    history_plot(profile, model_name, model_loc, hist_df, save_plot=True,
                    plot_file_name=f'history-{profile}-train.svg')
    history_plot(profile, model_name, model_loc, hist_df, save_plot=True,
                metrics=['mae', 'val_mae',
                        'rmse', 'val_rms'],
                plot_file_name=f'history-{profile}-valid.svg')
    
    #! Plot history of cycles
    # history_plot(profile, model_name, model_loc, hist_df, save_plot=True,
    #             metrics=['mae', 'val_mae',
    #                     'rmse', 'val_rms'],
    #             plot_file_name=f'history-{profile}-valid.svg')

    pd.DataFrame(TRAIN[4]).to_csv(f'{model_loc}{iEpoch}-train-logits.csv')
    pd.DataFrame(PERF[4]).to_csv(f'{model_loc}{iEpoch}-valid-logits.csv')
    pd.DataFrame(np.append(TEST1[4], TEST2[4])
                ).to_csv(f'{model_loc}{iEpoch}-test--logits.csv')
    
    # Reset every metric and clear memory leak
    MAE.reset_states()
    RMSE.reset_states()
    # RSquare.reset_states()
    # CR_ER.reset_states()
    
    # Flush and clean
    print('\n', flush=True)
    # tf.keras.backend.clear_session()
# %% [Running performance tests]
# test_S = 3000
# test_N = 8000
# x_test = x_train[test_S:test_S+test_N,:,:]
# y_test = y_train[test_S:test_S+test_N,:]
# # plt.plot(y_test[:,0])
# VIT_input = x_test[0,:,:3]
# SOC_input = x_test[0,:,3:]
# # VIT_input = np.ones(shape=x_test[0,:,:3].shape)*x_test[-1,0,:3]
# # SOC_input = np.ones(shape=x_test[0,:,3:].shape)*0.60#x_test[-1,,3:]
# PRED = np.zeros(shape=(y_test.shape[0],), dtype=np.float32)
# for i in trange(y_test.shape[0]):
#     logits = lstm_model.predict(
#                             x=np.expand_dims(
#                                 np.concatenate(
#                                     (VIT_input, SOC_input),
#                                     axis=1),
#                                 axis=0),
#                             batch_size=1
#                         )
#     VIT_input = x_test[i,:,:3]
#     SOC_input = np.concatenate(
#                         (SOC_input, np.expand_dims(logits,axis=0)),
#                         axis=0)[1:,:]
#     PRED[i] = logits
# MAE = np.mean(tf.keras.backend.abs(y_test[::,-1]-PRED))
# RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
#             y_test[::,-1]-PRED)))
# test_time = np.linspace(0, PRED.shape[0]/60, PRED.shape[0])
# def format_func(value, _):
#     return int(value*100)

# fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
# ax1.plot(test_time, y_test[:,-1], '-', label="Actual")
# ax1.plot(test_time, PRED, '--', label="Prediction")
# ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
# ax1.set_xlabel("Time Slice (min)", fontsize=32)
# ax1.set_ylabel("SoC (%)", fontsize=32)
# #if RMS_plot:
# ax2 = ax1.twinx()
# ax2.plot(test_time,
#     RMS,
#     label="RMS error", color='#698856')
# ax2.fill_between(test_time,
#     RMS,
#     color='#698856')
# ax2.set_ylabel('Error', fontsize=32, color='#698856')
# ax2.tick_params(axis='y', labelcolor='#698856', labelsize=24)
# ax2.set_ylim([-0.1,1.6])
# ax1.set_title(
#     #f"{file_name} {model_type}. {profile}-trained",
#     f"Initial Discharging State. Constant 71% charge",
#     fontsize=36)
# ax1.legend(prop={'size': 32})
# ax2.legend(loc='center right', bbox_to_anchor=(1.0,0.80), prop={'size': 32})
# ax1.tick_params(axis='both', labelsize=24)
# ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
# ax1.set_ylim([-0.1,1.2])
# fig.tight_layout()
# textstr = '\n'.join((
#     '$MAE  = {0:.2f}%$'.format(np.mean(MAE)*100, ),
#     '$RMSE = {0:.2f}%$'.format(np.mean(RMS)*100, )
#     # '$R2  = nn.nn%$'
#         ))
# ax1.text(0.66, 0.74, textstr, transform=ax1.transAxes, fontsize=30,
#     verticalalignment='top',
#     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
# fig.savefig(f'{model_loc}images/train-i71percent.svg')
# %%
# VIT_input = x_test_one[0,:,:3]
# SOC_input = x_test_one[0,:,3:]
# PRED = np.zeros(shape=(y_test_one.shape[0],), dtype=np.float32)
# for i in trange(y_test_one.shape[0]):
#     logits = lstm_model.predict(
#                             x=np.expand_dims(
#                                 np.concatenate(
#                                     (VIT_input, SOC_input),
#                                     axis=1),
#                                 axis=0),
#                             batch_size=1
#                         )
#     VIT_input = x_test_one[i,:,:3]
#     SOC_input = np.concatenate(
#                         (SOC_input, np.expand_dims(logits,axis=0)),
#                         axis=0)[1:,:]
#     PRED[i] = logits
# MAE = np.mean(tf.keras.backend.abs(y_test_one[::,-1]-PRED))
# RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
#             y_test_one[::,-1]-PRED)))

# # Time range
# test_time = np.linspace(0, PRED.shape[0]/60, PRED.shape[0])
# def format_func(value, _):
#     return int(value*100)

# fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
# ax1.plot(test_time, y_test_one[:,-1], '-', label="Actual")
# ax1.plot(test_time, PRED, '--', label="Prediction")
# # ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
# ax1.set_xlabel("Time Slice (min)", fontsize=32)
# ax1.set_ylabel("SoC (%)", fontsize=32)
# #if RMS_plot:
# ax2 = ax1.twinx()
# ax2.plot(test_time,
#     RMS,
#     label="RMS error", color='#698856')
# ax2.fill_between(test_time,
#     RMS,
#     color='#698856')
# ax2.set_ylabel('Error', fontsize=32, color='#698856')
# ax2.tick_params(axis='y', labelcolor='#698856', labelsize=28)
# ax2.set_ylim([-0.1,1.6])
# ax1.set_title(
#     #f"{file_name} {model_type}. {profile}-trained",
#     f"4-feature Model №2 LSTM FF {profile}-tr over DST set. {out_steps}-steps",
#     fontsize=36)
# ax1.legend(prop={'size': 32})
# ax2.legend(loc='center right', bbox_to_anchor=(1.0,0.80), prop={'size': 32})
# ax1.tick_params(axis='both', labelsize=28)
# ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
# ax1.set_ylim([-0.1,1.2])
# fig.tight_layout()
# textstr = '\n'.join((
#     '$MAE  = {0:.2f}%$'.format(np.mean(MAE)*100, ),
#     '$RMSE = {0:.2f}%$'.format(np.mean(RMS)*100, )
#     # '$R2  = nn.nn%$'
#         ))
# ax1.text(0.66, 0.74, textstr, transform=ax1.transAxes, fontsize=30,
#     verticalalignment='top',
#     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
# fig.savefig(f'{model_loc}SMR{profile}-Test One-{iEpoch}.svg')
# %%
# VIT_input = x_test_two[0,:,:3]
# SOC_input = x_test_two[0,:,3:]
# PRED = np.zeros(shape=(y_test_two.shape[0],), dtype=np.float32)
# for i in trange(y_test_two.shape[0]):
#     logits = lstm_model.predict(
#                             x=np.expand_dims(
#                                 np.concatenate(
#                                     (VIT_input, SOC_input),
#                                     axis=1),
#                                 axis=0),
#                             batch_size=1
#                         )
#     VIT_input = x_test_two[i,:,:3]
#     SOC_input = np.concatenate(
#                         (SOC_input, np.expand_dims(logits,axis=0)),
#                         axis=0)[1:,:]
#     PRED[i] = logits
# MAE = np.mean(tf.keras.backend.abs(y_test_two[::,-1]-PRED))
# RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
#             y_test_two[::,-1]-PRED)))

# # Time range
# test_time = np.linspace(0, PRED.shape[0]/60, PRED.shape[0])
# def format_func(value, _):
#     return int(value*100)

# fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
# ax1.plot(test_time, y_test_two[:,-1], '-', label="Actual")
# ax1.plot(test_time, PRED, '--', label="Prediction")
# # ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
# ax1.set_xlabel("Time Slice (min)", fontsize=32)
# ax1.set_ylabel("SoC (%)", fontsize=32)
# #if RMS_plot:
# ax2 = ax1.twinx()
# ax2.plot(test_time,
#     RMS,
#     label="RMS error", color='#698856')
# ax2.fill_between(test_time,
#     RMS,
#     color='#698856')
# ax2.set_ylabel('Error', fontsize=32, color='#698856')
# ax2.tick_params(axis='y', labelcolor='#698856', labelsize=24)
# ax2.set_ylim([-0.1,1.6])
# ax1.set_title(
#     #f"{file_name} {model_type}. {profile}-trained",
#     # f"VITpSoC №1 - {profile}-trained over US06 set. {out_steps}-steps",
#     f"4-feature Model №2 LSTM FF {profile}-tr over US06 set. {out_steps}-steps",
#     fontsize=36)
# ax1.legend(prop={'size': 32})
# ax2.legend(loc='center right', bbox_to_anchor=(1.0,0.80), prop={'size': 32})
# ax1.tick_params(axis='both', labelsize=24)
# ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
# ax1.set_ylim([-0.1,1.2])
# fig.tight_layout()
# textstr = '\n'.join((
#     '$MAE  = {0:.2f}%$'.format(np.mean(MAE)*100, ),
#     '$RMSE = {0:.2f}%$'.format(np.mean(RMS)*100, )
#     # '$R2  = nn.nn%$'
#         ))
# ax1.text(0.66, 0.74, textstr, transform=ax1.transAxes, fontsize=30,
#     verticalalignment='top',
#     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
# # fig.savefig(f'{model_loc}images/test_twoFull2.svg')
# fig.savefig(f'{model_loc}SMR{profile}-Test Two-{iEpoch}.svg')
# %%
# [Measuring performance of all models]
# file_name : str = os.path.basename(__file__)[:-3]
# model_loc : str = f'Models/{file_name}-{out_steps}steps/{profile}-models/'
# firstTime = False
# #* Run first 10 on 1 GPU, and remaining on another
# for j in range(17,21): #? Up to 21
#     model : AutoFeedBack = AutoFeedBack(units=510,
#         out_steps=out_steps, num_features=1
#     )
#     model.load_weights(f'{model_loc}{j}/{j}')
#     VIT_input = x_valid[0,:,:3]
#     SOC_input = x_valid[0,:,3:]
#     PRED = np.zeros(shape=(y_valid.shape[0],), dtype=np.float32)
#     for i in trange(y_valid.shape[0]):
#         logits = model.predict(
#                                 x=np.expand_dims(
#                                     np.concatenate(
#                                         (VIT_input, SOC_input),
#                                         axis=1),
#                                     axis=0),
#                                 batch_size=1
#                             )
#         VIT_input = x_valid[i,:,:3]
#         SOC_input = np.concatenate(
#                             (SOC_input, np.expand_dims(logits,axis=0)),
#                             axis=0)[1:,:]
#         PRED[i] = logits
#     FUDS_MAE = np.mean(tf.keras.backend.abs(y_valid[::,-1]-PRED))
#     FUDS_RMS = tf.keras.backend.sqrt(
#                     tf.keras.backend.mean(
#                         tf.keras.backend.square(
#                                 y_valid[::,-1]-PRED
#                             )
#                         )
#                     )
#     #! 3) Measure for 2nd
#     VIT_input = x_test_one[0,:,:3]
#     SOC_input = x_test_one[0,:,3:]
#     PRED = np.zeros(shape=(y_test_one.shape[0],), dtype=np.float32)
#     for i in trange(y_test_one.shape[0]):
#         logits = model.predict(
#                                 x=np.expand_dims(
#                                     np.concatenate(
#                                         (VIT_input, SOC_input),
#                                         axis=1),
#                                     axis=0),
#                                 batch_size=1
#                             )
#         VIT_input = x_test_one[i,:,:3]
#         SOC_input = np.concatenate(
#                             (SOC_input, np.expand_dims(logits,axis=0)),
#                             axis=0)[1:,:]
#         PRED[i] = logits
#     DST_MAE = np.mean(tf.keras.backend.abs(y_test_one[::,-1]-PRED))
#     DST_RMS = tf.keras.backend.sqrt(
#                     tf.keras.backend.mean(
#                         tf.keras.backend.square(
#                                 y_test_one[::,-1]-PRED
#                             )
#                         )
#                     )
#     #! 4) Measuire for 3rd
#     VIT_input = x_test_two[0,:,:3]
#     SOC_input = x_test_two[0,:,3:]
#     PRED = np.zeros(shape=(y_test_two.shape[0],), dtype=np.float32)
#     for i in trange(y_test_two.shape[0]):
#         logits = model.predict(
#                                 x=np.expand_dims(
#                                     np.concatenate(
#                                         (VIT_input, SOC_input),
#                                         axis=1),
#                                     axis=0),
#                                 batch_size=1
#                             )
#         VIT_input = x_test_two[i,:,:3]
#         SOC_input = np.concatenate(
#                             (SOC_input, np.expand_dims(logits,axis=0)),
#                             axis=0)[1:,:]
#         PRED[i] = logits
#     US_MAE = np.mean(tf.keras.backend.abs(y_test_two[::,-1]-PRED))
#     US_RMS = tf.keras.backend.sqrt(
#                     tf.keras.backend.mean(
#                         tf.keras.backend.square(
#                                 y_test_two[::,-1]-PRED
#                             )
#                         )
#                     )
#     #! 2) Save to df new values
#     result_df = pd.DataFrame(data={
#             'FUDS_MAE' : [FUDS_MAE],
#             'FUDS_RMS' : np.array(FUDS_RMS),

#             'DST_MAE'  : [DST_MAE],
#             'DST_RMS'  : np.array(DST_RMS),

#             'US06_MAE' : [US_MAE],
#             'US06_RMS' : np.array(US_RMS)
#         })
#     print(f'\nIteration: {j}', flush=True)
#     print(result_df)
#     with open(f'{model_loc}performance-{profile}.csv', mode='a') as f:
#         if(firstTime):
#             result_df.to_csv(f, index=False)
#             firstTime = False
#         else:
#             result_df.to_csv(f, index=False, header=False)
#     del model
#     plt.plot(y_valid[:,0])
#     plt.plot(PRED)
#     plt.show()
# %%
# perf_df = pd.read_csv(f'{model_loc}performance-{profile}.csv')
# best_v = [7, 9]

# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(28,12), dpi=600)
# ax1.plot(perf_df['FUDS_MAE']*100, '-', label='Training FUDS')
# ax1.plot(perf_df['DST_MAE']*100, '--', label='Validating DST')
# ax1.plot(perf_df['US06_MAE']*100, '-x', label='Validating US06')
# ax1.set_xlabel("Iterations/Epochs", fontsize=32)
# ax1.set_ylabel("MAE Error(%)", fontsize=32)  
# ax1.legend(prop={'size': 32})
# ax1.tick_params(axis='both', labelsize=28)
# ax1.plot(best_v, perf_df['FUDS_MAE'][best_v]*100, 'r*', markersize=32)
# ax1.set_ylim([0,15])

# ax2.plot(perf_df['FUDS_RMS']*100, '-', label='Training FUDS')
# ax2.plot(perf_df['DST_RMS']*100, '--', label='Validating DST')
# ax2.plot(perf_df['US06_RMS']*100, '-x', label='Validating US06')
# ax2.set_xlabel("Iterations/Epochs", fontsize=32)
# ax2.set_ylabel("RMSE Error(%)", fontsize=32)  
# ax2.legend(prop={'size': 32})
# ax2.tick_params(axis='both', labelsize=28)
# ax2.plot(best_v, perf_df['FUDS_RMS'][best_v]*100, 'r*', markersize=32)
# ax2.set_ylim([0,15])

# fig.suptitle('Accuracy evalution over training FUDS profile and validating with DST and US06', fontsize=32)
# fig.tight_layout()
# fig.savefig(f'{model_loc}SMR{profile}-performance.svg')

# %%
# optimiser = tf.optimizers.Adam(learning_rate = 0.0001)
# loss_fn   = tf.losses.MeanAbsoluteError()
# MAE     = tf.metrics.MeanAbsoluteError()
# RMSE    = tf.metrics.RootMeanSquaredError()
# # %%
# sh_i = np.arange(y_train.shape[0])
# for i in sh_i[:1]:
#     with tf.GradientTape() as tape:
#         logits     = lstm_model(x_train[i:i+1,:,:], training=True)
#         loss_value = loss_fn(y_train[i:i+1,:], logits)
#     grads = tape.gradient(loss_value, lstm_model.trainable_weights)
#     # optimiser.update_loss(prev_loss, loss_value)
#     optimiser.apply_gradients(zip(grads, lstm_model.trainable_weights))
#     #MAE.update_state(y_true=y[:1], y_pred=logits)
#     #RMSE.update_state(y_true=y[:], y_pred=logits)

# %%
# optimiser = tf.optimizers.Adam(learning_rate = 0.001)
# loss_fn   = tf.losses.MeanAbsoluteError()
# @tf.function
# def train_single_st(x, y, prev_loss):
#     with tf.GradientTape() as tape:
#         logits     = lstm_model(x, training=True)
#         loss_value = loss_fn(y, logits)
#     grads = tape.gradient(loss_value, lstm_model.trainable_weights)
#     # optimiser.update_loss(prev_loss, loss_value)
#     optimiser.apply_gradients(zip(grads, lstm_model.trainable_weights))
#     MAE.update_state(y_true=y[:1], y_pred=logits)
#     RMSE.update_state(y_true=y[:], y_pred=logits)
#     # RSquare.update_state(y_true=y[:], y_pred=logits)
#     return loss_value

# @tf.function
# def test_step(x):
#     return lstm_model(x, training=False)

# def valid_step(x, y):
#     logits  = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
#     loss    = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
#     mae     = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
#     rmse    = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
#     rsquare = np.zeros(shape=(y.shape[0], ), dtype=np.float32)
#     for i in trange(y.shape[0]):
#         logits[i] = test_step(x[i:i+1,:,:])
#         MAE.update_state(y_true=y[i:i+1], y_pred=logits[i])
#         RMSE.update_state(y_true=y[i:i+1], y_pred=logits[i])
#         # RSquare.update_state(y_true=y[i:i+1], y_pred=logits[i])
#         loss[i]    = loss_fn(y[i:i+1], logits[i])
#         mae[i]     = MAE.result()
#         rmse[i]    = RMSE.result()
#         # rsquare[i] = RSquare.result()
#     return [np.mean(loss), np.mean(mae), np.mean(rmse), np.mean(rsquare), logits]

# MAE     = tf.metrics.MeanAbsoluteError()
# RMSE    = tf.metrics.RootMeanSquaredError()
# # RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)
# pbar = tqdm(total=y_train.shape[0])
# sh_i = np.arange(y_train.shape[0])
# # np.random.shuffle(sh_i)
# for i in sh_i[:]:
#     loss_value = train_single_st(x_train[i:i+1,:,:], y_train[i:i+1,:],
#                                 None)
#     # Progress Bar
#     pbar.update(1)
#     pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
#                         f'loss: {loss_value:.4f} - '
#                         f'mae: {MAE.result():.4f} - '
#                         f'rmse: {RMSE.result():.4f} - '
#                         # f'rsquare: {RSquare.result():.4f}'
#                         )
# pbar.close()
# lstm_model.reset_metrics()
# lstm_model.reset_states()
# logits  = np.zeros(shape=(y_valid.shape[0], ), dtype=np.float32)
# for i in trange(y_valid.shape[0]):
#     logits[i] = test_step(x_valid[i:i+1,:,:])[-1]

# %%
# lstm_model.compile(loss=tf.losses.MeanAbsoluteError(),
#             optimizer=tf.optimizers.Adam(learning_rate = 0.0001),
#             metrics=[tf.metrics.MeanAbsoluteError(),
#                      tf.metrics.RootMeanSquaredError(),
#                     #  tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)
#                     ])
# while iEpoch < mEpoch:
#     iEpoch+=1
#     print(f"Epoch {iEpoch}/{mEpoch}")
    
#     train_hist = lstm_model.fit(
#                     x=x_train, y=y_train, epochs=1,
#                     validation_data=(x_valid, y_valid),
#                     callbacks=[early_stopping, nanTerminate],
#                     batch_size=1, shuffle=False
#                     #run_eagerly=True
#                 )
#     lstm_model.save(filepath=f'{model_loc}{iEpoch}',
#                         overwrite=True, include_optimizer=True,
#                         save_format='tf', signatures=None, options=None,
#                         save_traces=True
#                 )
#     if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
#         os.remove(f'{model_loc}{iEpoch-1}.ch')
#     os.mknod(f'{model_loc}{iEpoch}.ch')

#     # Saving history variable
#     # convert the history.history dict to a pandas DataFrame:     
#     hist_df = pd.DataFrame(train_hist.history)
#     # or save to csv:
#     with open(f'{model_loc}history-{profile}.csv', mode='a') as f:
#         if(firstLog):
#             hist_df.to_csv(f, index=False)
#             firstLog = False
#         else:
#             hist_df.to_csv(f, index=False, header=False)
    
#     #! Run the Evaluate function
#     #! Replace with tf.metric function.
#     PRED = lstm_model.predict(x_valid, batch_size=1)
#     RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
#                 y_valid[::,]-PRED)))
#     PERF = lstm_model.evaluate(x=x_valid,
#                                y=y_valid,
#                                batch_size=1,
#                                verbose=0)
#     # otherwise the right y-label is slightly clipped
#     predicting_plot(profile=profile, file_name='myModel №2',
#                     model_loc=model_loc,
#                     model_type='LSTM Test - Train dataset',
#                     iEpoch=f'val-{iEpoch}',
#                     Y=y_valid,
#                     PRED=PRED,
#                     RMS=RMS,
#                     val_perf=PERF,
#                     TAIL=y_valid.shape[0],
#                     save_plot=True)
#     if(PERF[-2] <=0.024): # Check thr RMSE
#         print("RMS droped around 2.4%. Breaking the training")
#         break
# %%
# # %%
# lstm_model.build((1,500,4))
# lstm_model.save(filepath=f'{model_loc}test/12-2',
#                     overwrite=True, include_optimizer=True,
#                     save_format='tf', signatures=None, options=None,
#                     save_traces=True
#             )
# # Convert the model to Tensorflow Lite and save.
# with open(f'{model_loc}myModel-№2-{profile}.tflite', 'wb') as f:
#     f.write(
#         # tf.lite.TFLiteConverter.from_saved_model(
#         #         saved_model_dir=f'{model_loc}19-tf'
#         #     ).convert()
#         # )
#         tf.lite.TFLiteConverter.from_keras_model(
#                 model=test_lstm_model
#             ).convert()
#         )