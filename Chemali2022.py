#!/usr/bin/python
# %% [markdown]
# # # 1
# # #
# # LSTM for SoC by Ephrem Chemali 2022
# Modefied version
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

from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
from cy_modules.utils import str2bool
from py_modules.plotting import predicting_plot

from scipy import integrate # integration with trapizoid
from tqdm import trange
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

opts = [('-d', 'False'), ('-e', '50'), ('-g', '0'), ('-p', 'FUDS')]
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
    VIT : tf.keras.models.Sequential = tf.keras.models.load_model(
            filepath='Models/Chemali2017/FUDS-models/48', compile=True,
            custom_objects={"RSquare": tfa.metrics.RSquare,
                            "custom_loss": custom_loss}
        )

except:
    print('One of the models failed to load.')
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
# BMSid   = 0
# cell = 1
# Data = BMSsData[BMSid][['Current(A)', f'6-Cell_{cell}', f'Sns_{cell}']].to_numpy()

# for cell in range(2, 11):
#     Data = np.append(
#             Data,
#             BMSsData[BMSid][['Current(A)', f'6-Cell_{cell}', f'Sns_{cell}']].to_numpy(),
#             axis=0
#         )

# # newCC = CC[-1,bms] + 
# # test = 0.98 + ccSoC(current=BMSsData[BMSid]['Current(A)'].to_numpy(), 
# #       time_s=BMSsData[BMSid]['Cycle_Time(s)'].to_numpy())
# # plt.plot(test)
# for BMSid in range(1, 6):
#     for cell in range(1, 11):
#         Data = np.append(
#                 Data,
#                 BMSsData[BMSid][['Current(A)', f'6-Cell_{cell}', f'Sns_{cell}']].to_numpy(),
#                 axis=0
#             )

# %%

n_samples = (BMSsData[0].shape[0]*10*6)-(500*6*10)
bms_samples = BMSsData[BMSid].shape[0]-500
cell_samples = 10
X_windows = np.zeros(shape=(n_samples,500,3), dtype=np.float32)
Y_windows = np.zeros(shape=(n_samples,1), dtype=np.float32)
for BMSid in range(0, 6):
    print(f'BMS id: {BMSid}')
    for cell in range(1, 11):
        print(f'Cell: {cell}')
        for i in range(0,bms_samples):
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
            X_windows[index,:,:] = input_set
            
            # BMSsData[BMSid].loc[i:499+i,
            #         ['Current(A)', f'6-Cell_{cell}', f'Sns_{cell}']
            #     ].to_numpy()

            # Y_windows = np.append(Y_windows,
            #     np.expand_dims(BMSsData[BMSid].loc[499+i,
            #         ['SoC(%)']
            #     ].to_numpy(), axis=0),
            #     axis=0)
            Y_windows[index,:] = BMSsData[BMSid].loc[499+i,
                    [f'SoC_{cell}(%)']
                ].to_numpy()
print(f'Data windowing ready')
# %%

# VIT.compile(loss=custom_loss,
#         optimizer=tf.optimizers.Adam(learning_rate=0.001,
#                 beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
#         metrics=[tf.metrics.MeanAbsoluteError(),
#                     tf.metrics.RootMeanSquaredError(),
#                     tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
#         #run_eagerly=True
#     )
#! CHeck NaN presense
print('X-Nans')
print(np.where(np.isnan(X_windows[:, :, :])))
print('Y-Nans')
print(np.where(np.isnan(Y_windows[:, :])))
nanTerminate = tf.keras.callbacks.TerminateOnNaN()
# %%
#! Normalise data before running
# for i in trange(0, X_windows.shape[0]):
#     history = VIT.fit(x=X_windows[i:i+1, :, :],
                        
#                       y=Y_windows[i:i+1,:],
#                         epochs=1,
#                         # validation_data=(x_valid, y_valid),
#                         callbacks=[nanTerminate],
#                         batch_size=1, shuffle=False, verbose=0
                        # )
file_name : str = os.path.basename(__file__)[:-3]
model_loc : str = f'Models/{file_name}/{profile}-models/'
# %%
iEpoch = 1
history = VIT.fit(x=X_windows[:, :, :],
                        
                  y=Y_windows[:,:],
                        epochs=1,
                        # validation_data=(x_valid, y_valid),
                        callbacks=[nanTerminate],
                        batch_size=1, shuffle=True #!Change back to true
                        )
#! Model saving to somewhere
VIT.save(filepath=f'{model_loc}{iEpoch}',
                            overwrite=True, include_optimizer=True,
                            save_format='h5', signatures=None, options=None,
                            save_traces=True
                )
# Convert the model to Tensorflow Lite and save.
with open(f'{model_loc}postModel-№1-{profile}.tflite', 'wb') as f:
    f.write(
        tf.lite.TFLiteConverter.from_keras_model(
                model=VIT
            ).convert()
        )
    # f.write(
    #     tf.lite.TFLiteConverter.from_saved_model(
    #             f'{model_loc}{iEpoch}'
    #         ).convert()
    #     )
# %%
test = VIT.predict(x=X_windows[:8000, :, :], batch_size=1, verbose=1)
plt.plot(test)
plt.plot(Y_windows[:8000,:])
