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
from tqdm import trange

from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
from py_modules.AutoFeedBack import AutoFeedBack
from py_modules.RobustAdam import RobustAdam
from cy_modules.utils import str2bool
from py_modules.plotting import predicting_plot# %%


from scipy import integrate # integration with trapizoid
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

opts = [('-d', 'False'), ('-e', '1'), ('-g', '0'), ('-p', 'FUDS'), ('-s', '30')]
mEpoch    : int = 10
GPU       : int = 0
profile   : str = 'DST'
out_steps : int = 10
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
    elif opt in ("-s", "--steps"):
        out_steps = int(arg)
# %%
# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['font.family'] = 'Bender'

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
def custom_loss(y_true, y_pred):
    """ Custom loss based on following formula:
        sumN(0.5*(SoC-SoC*)^2)

    Args:
        y_true (tf.Tensor): True output values
        y_pred (tf.Tensor): Predicted output from model

    Returns:
        tf.Tensor: The calculated Loss Value
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    loss = tf.math.divide(
                        x=tf.keras.backend.square(
                                x=tf.math.subtract(x=y_true,
                                                   y=y_pred)
                            ), 
                        y=2
                    )
    return tf.keras.backend.sum(loss, axis=1)
try:
    print("Using Post-trained model")
    VIT : tf.keras.models.Sequential = tf.keras.models.load_model(
        filepath='Models/Chemali2021/FUDS-models/1', compile=False,
        custom_objects={"RSquare": tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32),
                        "custom_loss": custom_loss}
    )
    VIT.compile(loss=custom_loss,
        optimizer=tf.optimizers.Adam(learning_rate=0.001,
                beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
        metrics=[tf.metrics.MeanAbsoluteError(),
                    tf.metrics.RootMeanSquaredError(),
                    tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
        #run_eagerly=True
    )
except:
    print('Using pretrained model')
    VIT : tf.keras.models.Sequential = tf.keras.models.load_model(
        filepath='Models/Chemali2017/FUDS-models/48', compile=True,
        custom_objects={"RSquare": tfa.metrics.RSquare,
                        "custom_loss": custom_loss}
    )
MEAN = np.array([-0.35640615,  3.2060466 , 30.660755  ], dtype=np.float32)
STD  = np.array([ 0.9579658 ,  0.22374259, 13.653275  ], dtype=np.float32)
# %%
def ccSoC(current   : pd.Series,
          time_s    : pd.Series,
          n_capacity: float = 2.5 ) -> pd.Series:
    """ Return SoC based on Couloumb Counter.
    @ 25deg I said it was 2.5

    Args:
        chargeData (pd.Series): Charge Data Series
        discargeData (pd.Series): Discharge Data Series

    Raises:
        ValueError: If any of data has negative
        ValueError: If the data trend is negative. (end-beg)<0.

    Returns:
        pd.Series: Ceil data with 2 decimal places only.
    """
    return (1/(3600*n_capacity))*(
            integrate.cumtrapz(
                    y=current.to_numpy(), x=time_s.to_numpy(),
                    dx=1, axis=-1, initial=0
                )
        )
output_loc  : str = 'Data/BMS_data/July09-FUDS1/'
BMSsData = []
for BMSid in range(0, 6):
    try:
        #* Voltages
        BMSsData.append(
                pd.read_csv(filepath_or_buffer=f'{output_loc}Filt_CANid_{BMSid}.csv',
                            sep=',', verbose=True)
            )
        print(f'Cycle samples of V at BMS{BMSid} - {BMSsData[BMSid].shape} and T -{BMSsData[BMSid].shape}')
        #! Get initial SoC with model
        # initial_SoC = np.zeros(shape=(10,), dtype=np.float32)
        for cell in range(1,11):
            #* Get values
            input_set = np.expand_dims(np.divide(
                            np.subtract(
                                    BMSsData[BMSid].loc[:499,
                                            ['Current(A)', f'6-Cell_{cell}', f'Sns_{cell}']
                                        ].to_numpy(),
                                    MEAN
                                ),
                            STD
                        ),axis=0)
            #* Eliminate NaN
            location = np.where(np.isnan(input_set))
            # print(location)
            input_set[location] = input_set[location[0],location[1]-1,location[2]]
            initial_SoC = np.round(VIT.predict(input_set)[0][0], decimals=2)
            #* Fill the remaining data
            BMSsData[BMSid].loc[:499, f'SoC_{cell}(%)'] = initial_SoC
            BMSsData[BMSid].loc[500:, f'SoC_{cell}(%)'] = np.round(initial_SoC +\
                ccSoC(current = BMSsData[BMSid].loc[500:,'Current(A)'], 
                    time_s  = BMSsData[BMSid].loc[500:,'Cycle_Time(s)']), decimals=2)
    except Exception as e:
        print(e)
        print(f'============Failed to extract Cycle data of BMS {BMSid}==============')

# %%
n_samples = (BMSsData[0].shape[0]*10*6)-(500*6*10)
bms_samples = BMSsData[BMSid].shape[0]-500
cell_samples = 10
X_windows = np.zeros(shape=(n_samples,500,4), dtype=np.float32)
Y_windows = np.zeros(shape=(n_samples,out_steps), dtype=np.float32)
#! You can speed it app to add new columns - prevSoC with a shift 1. You did it
#!once with old code, it should not be difficult. Then no concatination needed.
for BMSid in range(0, 2):   #!6
    print(f'BMS id: {BMSid}')
    for cell in range(1, 11):   #! 11
        print(f'Cell: {cell}')
        for i in range(1,bms_samples):  #! Start from 1
            # X_windows = np.append(X_windows,
            #     np.expand_dims(BMSsData[BMSid].loc[i:499+i,
            #         ['Current(A)', f'6-Cell_{cell}', f'Sns_{cell}']
            #     ].to_numpy(), axis=0),
            #     axis=0)
            # index = BMSid*cell_samples*bms_samples*(cell-1)+i
            index = (BMSid*bms_samples*cell_samples)+(bms_samples*(cell-1))+i
            # X_windows[index,:,:] = BMSsData[BMSid].loc[i:499+i,
            #         ['Current(A)', f'6-Cell_{cell}', f'Sns_{cell}']
            #     ].to_numpy()
            input_set = np.divide(
                            np.subtract(
                                    BMSsData[BMSid].loc[i:499+i,
                                            ['Current(A)', f'6-Cell_{cell}', f'Sns_{cell}']
                                        ].to_numpy(),
                                    MEAN
                                ),
                            STD
                        )
            location = np.where(np.isnan(input_set))
            if len(location[0]) > 0:
                for j in range(len(location[0])):
                    input_set[location[0][j], location[1][j]] = \
                        input_set[location[0][j]-1,location[1][j]]
            #! Add the SoC to the input
            input_soc = BMSsData[BMSid].loc[i-1:i+498,
                    [f'SoC_{cell}(%)']
                ].to_numpy()
            X_windows[index,:,:] = np.concatenate((input_set, input_soc), axis=1)

            Y_windows[index,:] = BMSsData[BMSid].loc[500+i-out_steps:499+i,
                    [f'SoC_{cell}(%)']
                ].to_numpy()[:,0]
print(f'Data windowing ready')
#? CHeck NaN presense
print('X-Nans')
print(np.where(np.isnan(X_windows[:, :, :])))
print('Y-Nans')
print(np.where(np.isnan(Y_windows[:, :])))
# %%
file_name : str = os.path.basename(__file__)[:-3]
origin_model_loc : str = f'Models/Sadykov2021-{out_steps}steps/{profile}-models/'
model_loc : str = f'Models/{file_name}-{out_steps}steps/{profile}-models/'
iEpoch = 0
firstLog  : bool = True
try:
    #! Get the best we trained from FUDS, based on error plot
    model : AutoFeedBack = AutoFeedBack(units=510,
            out_steps=out_steps, num_features=1
        )
    model.load_weights(f'{origin_model_loc}7/7')    #* 7 or 9
    firstLog = False
    print("Original model obtained")
except:
    print("Original model faced issues with obtaining")

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        min_delta=0, patience=2, verbose=0, mode='min',
        baseline=None, restore_best_weights=False
    )
nanTerminate = tf.keras.callbacks.TerminateOnNaN()

# %%
#! I changed learning rate from 0.001 to 0.0001 after first run. If further fails
#!replace back. The drop was present.
model.compile(loss=tf.losses.MeanAbsoluteError(),
            optimizer=tf.optimizers.Adam(learning_rate = 0.0001), #!Start: 0.001
            metrics=[tf.metrics.MeanAbsoluteError(),
                     tf.metrics.RootMeanSquaredError(),
                     tfa.metrics.RSquare(y_shape=(out_steps,), dtype=tf.float32)
                    ])
while iEpoch < mEpoch:
    iEpoch+=1
    #! 15 Hours!!! Configure early stop or use 2 bricks only and last 2 cells
    #!for validation
    train_hist = model.fit(
                    x=X_windows[:index, :, :], y=Y_windows[:index,:], epochs=1,
                    callbacks=[nanTerminate],
                    batch_size=1, shuffle=True
                )
    # Saving model
    model.save_weights(filepath=f'{model_loc}{iEpoch}/{iEpoch}',
            overwrite=True, save_format='tf', options=None
        )
    model.save(filepath=f'{model_loc}{iEpoch}-tf',
                            overwrite=True, include_optimizer=True,
                            save_format='tf', signatures=None, options=None,
                            save_traces=True
                )
    if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
        os.remove(f'{model_loc}{iEpoch-1}.ch')
    os.mknod(f'{model_loc}{iEpoch}.ch')

    # PERF = valid_step(x_valid, y_valid)
    # hist_df = pd.DataFrame(data={
    #         'loss'   : [np.array(loss_value)],
    #         'mae'    : [np.array(MAE.result())],
    #         'rmse'   : [np.array(RMSE.result())],
    #         'rsquare': [np.array(RSquare.result())]
    #     })
    # hist_df['vall_loss'] = PERF[0]
    # hist_df['val_mean_absolute_error'] = PERF[1]
    # hist_df['val_root_mean_squared_error'] = PERF[2]
    # hist_df['val_r_square'] = PERF[3]
    
    # or save to csv:
    hist_df = pd.DataFrame(train_hist.history)
    PERF = model.evaluate(x=X_windows[index-7000:index, :, :],
                          y=Y_windows[index-7000:index,:],
                          batch_size=1,
                          verbose=1)
    hist_df['vall_loss'] = PERF[0]
    hist_df['val_mean_absolute_error'] = PERF[1]
    hist_df['val_root_mean_squared_error'] = PERF[2]
    hist_df['val_r_square'] = PERF[3]

    with open(f'{model_loc}history-{profile}.csv', mode='a') as f:
        if(firstLog):
            hist_df.to_csv(f, index=False)
            firstLog = False
        else:
            hist_df.to_csv(f, index=False, header=False)
    
    PRED = model.predict(
                    x=X_windows[index-7000:index, :, :], verbose=1,
                    batch_size=1
                )
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                Y_windows[index-7000:index,-1]-PRED)))

    # otherwise the right y-label is slightly clipped
    predicting_plot(profile=profile, file_name='4-feature Model №2',
                    model_loc=model_loc,
                    model_type='LSTM Train',
                    iEpoch=f'val-{iEpoch}',
                    Y=Y_windows[index-7000:index,-1:],
                    PRED=PRED,
                    RMS=np.expand_dims(RMS,axis=1),
                    val_perf=PERF[:4],
                    TAIL=Y_windows[index-7000:index,:].shape[0],
                    save_plot=True,
                    RMS_plot=True) #! Saving memory from high errors.
    # otherwise the right y-label is slightly clipped
    if(PERF[-3] <=0.024): # Check thr RMSE
        print("RMS droped around 2.4%. Breaking the training")
        break

    VIT_input = X_windows[index-7000, :, :3]
                # x_valid[0,:,:3]
    SOC_input = X_windows[index-7000, :, 3:]
                # x_valid[0,:,3:]
    PRED = np.zeros(shape=(Y_windows[index-7000:index,:].shape[0],), dtype=np.float32)
    for i in trange(7000):
        logits = model.predict(
                                x=np.expand_dims(
                                    np.concatenate(
                                        (VIT_input, SOC_input),
                                        axis=1),
                                    axis=0),
                                batch_size=1
                            )
        VIT_input = X_windows[index-7000+i, :, :3] #x_valid[i,:,:3]
        SOC_input = np.concatenate(
                            (SOC_input, np.expand_dims(logits,axis=0)),
                            axis=0)[1:,:]
        PRED[i] = logits
    MAE = np.mean(tf.keras.backend.abs(Y_windows[index-7000:index,-1]-PRED))
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                Y_windows[index-7000:index,-1]-PRED)))
    # Time range
    test_time = np.linspace(0, PRED.shape[0]/60, PRED.shape[0])
    def format_func(value, _):
        return int(value*100)

    fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
    ax1.plot(test_time, Y_windows[index-7000:index,-1], '-', label="Actual")
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
        f"4-feature Model №2 LSTM Feed-forward Cycler,  {out_steps}-steps",
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