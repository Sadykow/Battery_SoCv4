# %%
import os
from typing import get_type_hints
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

from datetime import datetime
from matplotlib.lines import Line2D
from time import perf_counter
from tqdm import trange

sys.path.append(os.getcwd() + '/..')
from extractor.DataGenerator import *
from py_modules.Attention import *

float_dtype : type = np.float32
int_dtype   : type = np.int32

hmi_dir    : str  = '../Data/HMI_FILES/'
hmi_file   : str  = 'battery_test_log_14.csv'

bms_dir    : str  = '../Data/BMS_data/'
bms_file   : str  = 'FirstBalanceCharge.json'
#! Select GPU for usage. CPU versions ignores it.
#!! Learn to check if GPU is occupied or not.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
GPU=0
if physical_devices:
    #! With /device/GPU:1 the output was faster.
    #! need to research more why.
    tf.config.experimental.set_visible_devices(
                            physical_devices[GPU], 'GPU')

    #if GPU == 1:
    tf.config.experimental.set_memory_growth(
                            physical_devices[GPU], True)
    print("GPU found and memory growth enabled") 
    
    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    print("GPU found") 
    print(f"\nPhysical GPUs: {len(physical_devices)}"
                  f"\nLogical GPUs: {len(logical_devices)}")
#! For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float32')

# %%
#? Getting HMI data
tic : int = perf_counter()

df : pd.DataFrame = pd.read_csv(filepath_or_buffer=hmi_dir+hmi_file,
                sep='\n', delimiter=';',
                # Column and Index Locations and Names
                header='infer', names=None, index_col=None,
                usecols=None, squeeze=False, prefix=None, mangle_dupe_cols=True,
                # General Parsing Configuration
                dtype=None, engine='python', converters=None, true_values=None,
                false_values=None, skipinitialspace=False, skiprows=3,
                skipfooter=1, nrows=None,
                # NA and Missing Data Handling
                na_values=None, keep_default_na=True, na_filter=True,
                verbose=False, skip_blank_lines=True,
                # Datetime Handling            
                parse_dates=False, infer_datetime_format=False,
                keep_date_col=False, date_parser=None, dayfirst=False,
                cache_dates=True,
                # Iteration
                iterator=False, chunksize=None,
                # Quoting, Compression, and File Format
                compression='infer',
                thousands=None, decimal=',', lineterminator=None, quotechar='"',
                quoting=0, doublequote=True, escapechar=None, comment=None,
                encoding=None, dialect=None,
                # Error Handling
                error_bad_lines=True, warn_bad_lines=True,
                # Internal
                delim_whitespace=False, low_memory=True,
                memory_map=False, float_precision=None
            )
dates_list = [datetime.strptime(time, '%H:%M:%S,%f')
                for time in np.array(df['Time'])]
Data : pd.DataFrame = pd.DataFrame(data = {
    'Step_Time(s)': [date.microsecond/1000000 + date.second 
                     + date.minute*60 + date.hour*3600
                        for date in dates_list],
    'Voltage(V)' : np.array(
        object=df['Uactual'].str.replace(',', '.').str.replace('V', ''),
        dtype=np.float32),
    'Current(A)' : np.array(
        object=df['Iactual'].str.replace(',', '.').str.replace('A', ''),
        dtype=np.float32),
    'Power(W)'   : np.array(
        object=df['Pactual'].str.replace('W', ''),
        dtype=np.float32),
    'Discharge_Capacity(Ah)' : np.array(
        object=df['Ah'].str.replace(',', '.').str.replace('Ah', ''),
        dtype=np.float32),
    'Discharge_Power(Wh)' : np.array(
        object=df['Wh'].str.replace(',', '.').str.replace('Wh', ''),
        dtype=np.float32)
    })
print(f'Parsing HMI data took: {perf_counter()-tic}')
# %%
#? Getting BMS data
BMSVoltages     : np.ndarray = np.empty(shape=(1,10), dtype=np.float32)
BMSTemperatures : np.ndarray = np.empty(shape=(1,12), dtype=np.float32)
BMSsV = []
BMSsT = []
for i in range(6):
    BMSsV.append(BMSVoltages)
    BMSsT.append(BMSTemperatures)
tic : int = perf_counter()
with open(bms_dir+bms_file) as file_object:
    # store file data in object
    lines = file_object.readlines()
    c_lines = 0
    
    for line in lines[:]:
        record = line.replace(':', ' : ').replace(',', '').replace('{','').replace('}','').replace('[','').replace(']','').split()
        #print(record)
        try:
            if(record[0] == '"VoltageInfo"'):
                BMSsV[int(record[7])] = np.append(
                                            arr=BMSsV[int(record[7])],
                                            values=np.array([float(v) for v in record[10:20]], ndmin=2),
                                            axis=0)
            elif(record[0] == '"TemperatureInfo"'):
                BMSsT[int(record[7])] = np.append(
                                            arr=BMSsT[int(record[7])],
                                            values=np.array([float(v) for v in record[10:22]], ndmin=2),
                                            axis=0)
            else:
                print("Unattended field: "+record[0])
        except Exception as inst:
            print(f'Unusable record Line: {c_lines}')
        c_lines += 1
print(f'Parsing BMS data of {c_lines} lines took {perf_counter()-tic}')
for i in range(6):
    print(f"BMS:{i}: V:{BMSsV[i].shape} and T:{BMSsT[i].shape}")
# %%
#! Plotting all data
mpl.rcParams['figure.figsize'] = (16, 12)
mpl.rcParams['axes.grid'] = False
#? HMI data
# Voltage
plt.figure()
plt.plot(Data['Step_Time(s)'], Data['Voltage(V)'], 'r')
plt.title('Voltage', size=16)
plt.ylabel('V', size=16)
plt.xlabel('Time (s)', size=16)
plt.xticks(size=14)
plt.yticks(size=14)
plt.ylim([50,70])
plt.grid(which='major', alpha=1)

# Current
plt.figure()
plt.plot(Data['Step_Time(s)'], Data['Current(A)'], 'b')
plt.title('Current', size=16)
plt.ylabel('A', size=16)
plt.xlabel('Time (s)', size=16)
plt.xticks(size=14)
plt.yticks(size=14)
# plt.ylim([-20.5, -19.5])
#plt.xlim([1000, 1150])
plt.grid(which='major', alpha=1)

# Power
plt.figure()
plt.plot(Data['Step_Time(s)'], Data['Power(W)'], 'k')
plt.title('Power', size=16)
plt.ylabel('W', size=16)
plt.xlabel('Time (s)', size=16)
plt.xticks(size=14)
plt.yticks(size=14)
# plt.ylim([-1400, -1200])
plt.grid(which='major', alpha=1)

# Capacity
plt.figure()
plt.plot(Data['Step_Time(s)'], Data['Discharge_Capacity(Ah)'], 'b')
plt.title('Discharge_Capacity', size=16)
plt.ylabel('Ah', size=16)
plt.xlabel('Time (s)', size=16)
plt.xticks(size=14)
plt.yticks(size=14)
#plt.ylim([-1400, -1200])
plt.grid(which='major', alpha=1)

# Power
plt.figure()
plt.plot(Data['Step_Time(s)'], Data['Discharge_Power(Wh)'], 'k')
plt.title('Discharge Power', size=16)
plt.ylabel('Wh', size=16)
plt.xlabel('Time (s)', size=16)
plt.xticks(size=14)
plt.yticks(size=14)
#plt.ylim([-1400, -1200])
plt.grid(which='major', alpha=1)
# %%
# BMS 
cmap = plt.cm.rainbow


custom_lines = [Line2D([0], [0], color=cmap(0.1), lw=5),
                Line2D([0], [0], color=cmap(0.2), lw=5),
                Line2D([0], [0], color=cmap(0.3), lw=5),
                Line2D([0], [0], color=cmap(0.4), lw=5),
                Line2D([0], [0], color=cmap(0.5), lw=5),
                Line2D([0], [0], color=cmap(0.6), lw=5),
                Line2D([0], [0], color=cmap(0.7), lw=5),
                Line2D([0], [0], color=cmap(0.8), lw=5),
                Line2D([0], [0], color=cmap(0.9), lw=5),
                Line2D([0], [0], color=cmap(1.0), lw=5)
                ]
# Voltages
colormap_set = [cmap(0.1),cmap(0.2),cmap(0.3),cmap(0.4),cmap(0.5),
            cmap(0.6),cmap(0.7),cmap(0.8),cmap(0.9),cmap(1)]
fig, axs = plt.subplots(2, 3)
for i in range(len(colormap_set)):
    axs[0, 0].plot(BMSsV[0][1:,i], color=colormap_set[i])
    if (not any(i == t for t in [0, 1])):
        axs[0, 1].plot(BMSsV[1][1:,i], color=colormap_set[i])
        axs[0, 2].plot(BMSsV[2][1:,i], color=colormap_set[i])
        
        axs[1, 0].plot(BMSsV[3][1:,i], color=colormap_set[i])
        axs[1, 1].plot(BMSsV[4][1:,i], color=colormap_set[i])
        axs[1, 2].plot(BMSsV[5][1:,i], color=colormap_set[i])
axs[0, 0].set_title('BMS 1')
axs[0, 1].set_title('BMS 2')
axs[0, 2].set_title('BMS 3')
axs[1, 0].set_title('BMS 4')
axs[1, 1].set_title('BMS 5')
axs[1, 2].set_title('BMS 6')

i = 0
for ax in axs.flat:
    #ax.set(ylim=[2.6,3.7])
    ax.set(xlabel='Voltage (V)', ylim=[3.25,3.8])
    ax.grid(b=True, axis='both', linestyle='-', linewidth=1)
    if( any(i == t for t in [1, 2, 3, 4, 5])):
        ax.legend(custom_lines[2:], ['Sensor 3', 'Sensor 4',
                                'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8',
                                'Sensor 9', 'Sensor10'])
    else:
        ax.legend(custom_lines, ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4',
                             'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8',
                             'Sensor 9', 'Sensor10'])
    i += 1
# Temperatures
custom_linesT = [Line2D([0], [0], color=cmap(0.00), lw=5),
                 Line2D([0], [0], color=cmap(0.09), lw=5),
                 Line2D([0], [0], color=cmap(0.18), lw=5),
                 Line2D([0], [0], color=cmap(0.27), lw=5),
                 Line2D([0], [0], color=cmap(0.36), lw=5),
                 Line2D([0], [0], color=cmap(0.45), lw=5),
                 Line2D([0], [0], color=cmap(0.55), lw=5),
                 Line2D([0], [0], color=cmap(0.64), lw=5),
                 Line2D([0], [0], color=cmap(0.73), lw=5),
                 Line2D([0], [0], color=cmap(0.82), lw=5),
                 Line2D([0], [0], color=cmap(0.91), lw=5),
                 Line2D([0], [0], color=cmap(1.00), lw=5)
                ]
colormapT_set = [cmap(0.00),cmap(0.09),cmap(0.18),cmap(0.27),cmap(0.36),
                 cmap(0.45),cmap(0.55),cmap(0.64),cmap(0.73),cmap(0.82),
                 cmap(0.91), cmap(1.00)]

figT, axTs = plt.subplots(2, 3)
for i in range(len(colormapT_set)):
    if (not any(i == t for t in [6, 7, 8, 9, 10, 11])):
        axTs[0, 0].plot(BMSsT[0][1:,i], color=colormapT_set[i])
    axTs[0, 1].plot(BMSsT[1][1:,i], color=colormapT_set[i])
    if (not any(i == t for t in [0, 1, 2, 3, 4, 5, 8])):
        axTs[0, 2].plot(BMSsT[2][1:,i], color=colormapT_set[i])

    if (not any(i == t for t in [0, 1, 2, 3, 6, 7, 8])):
        axTs[1, 0].plot(BMSsT[3][1:,i], color=colormapT_set[i])
    axTs[1, 1].plot(BMSsT[4][1:,i], color=colormapT_set[i])
    axTs[1, 2].plot(BMSsT[5][1:,i], color=colormapT_set[i])
axTs[0, 0].set_title('BMS 1')
axTs[0, 1].set_title('BMS 2')
axTs[0, 2].set_title('BMS 3 - minor outliers')
axTs[1, 0].set_title('BMS 4')
axTs[1, 1].set_title('BMS 5')
axTs[1, 2].set_title('BMS 6 - unclear data')

i = 0
for axT in axTs.flat:
    axT.set(xlabel='Temperature (T)', ylim=[19.5,30])
    axT.grid(b=True, axis='both', linestyle='-', linewidth=1)
    if( i == 0):
        axT.legend(custom_linesT[:7],
                   ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4',
                             'Sensor 5', 'Sensor 6'])
    elif( i == 2):
        axT.legend(custom_linesT[6:11], ['Sensor 7', 'Sensor 8',
                             'Sensor10', 'Sensor11', 'Sensor12'])
    elif( i == 3):
        axT.legend(custom_linesT[9:], ['Sensor10', 'Sensor11', 'Sensor12'])
    else:
        axT.legend(custom_linesT,
                            ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4',
                             'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8',
                             'Sensor 9', 'Sensor10', 'Sensor11', 'Sensor12'])
    i += 1
# %%
# plt.plot(Data['Current(A)']/3) # 10A - 3 parallel by 2 bricks.
# 6 parallels inside single brick

#! 1) Get model. Get shape.
#!  1.1) Current(A), Voltage(V), Temperature(C)
#!  1.2) Input shape of all 500 by 3 in UP order.
#! 2) Get data model used for training and normolise by training set model came
#!      from. Try all 3.
#! 3) Try on single line with constant current.
#! 4) Plot single.
#! 5) Get everythin else
# %%
data_dir    : str = '../Data/'
dataGenerator = DataGenerator(train_dir=f'{data_dir}A123_Matt_Set',
                              valid_dir=f'{data_dir}A123_Matt_Val',
                              test_dir=f'{data_dir}A123_Matt_Test',
                              columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                                'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                                ],
                              PROFILE_range = profile)
MEAN = np.mean(a=dataGenerator.train[:,:3], axis=0,
                                    dtype=float_dtype,
                                    keepdims=False)
STD = np.std(a=dataGenerator.train[:,:3], axis=0,
                            keepdims=False)
# !! Compare with original
normalised_training = np.divide(
    np.subtract(
            np.copy(a=dataGenerator.train_list[0].iloc[:,:3]),
            MEAN
        ),
    STD
    )
# %%
#? Model №1 - Chemali2017    - DST  - 1
#?                           - US06 - 50
#?                           - FUDS - 48

#? Model №2 - BinXiao2020    - DST  - 50
#?                           - US06 - 2 (21)
#?                           - FUDS - 48

#? Model №3 - TadeleMamo2020 - DST  - ?
#?                           - US06 - 25
#?                           - FUDS - 10

#? Model №7 - WeiZhang2020   - DST  - 9
#?                           - US06 - ?
#?                           - FUDS - 3
author  : str = 'BinXiao2020'#'TadeleMamo2020'#'WeiZhang2020'#Chemali2017
profile : str = 'FUDS'#'FUDS'#'US06'#'DST'
iEpoch  : int = 48
model_loc : str = f'../Models/{author}/{profile}-models/'

try:
    # for _, _, files in os.walk(model_loc):
    #     for file in files:
    #         if file.endswith('.ch'):
    #             iEpoch = int(os.path.splitext(file)[0])
    
    # model : tf.keras.models.Sequential = tf.keras.models.load_model(
    #         f'{model_loc}{iEpoch}',
    #         compile=False,
    #         custom_objects={"RSquare": tfa.metrics.RSquare}
    #         )
    #! Mamo case
    model : tf.keras.models.Sequential = tf.keras.models.load_model(
            f'{model_loc}{iEpoch}',
            compile=False,
            custom_objects={"RSquare": tfa.metrics.RSquare,
                            "AttentionWithContext": AttentionWithContext,
                            "Addition": Addition,
                            }
            )
    firstLog = False
    print("Model Identefied.")
except OSError as identifier:
    print("Model Not Found, Check the path. {} \n".format(identifier))


#? Given: 28408 samples, 1s => 7.89h
#?                       10s => +-48min
fs = 10
BMS_id = 0
cell = 0 # 0-9=
length = BMSsV[BMS_id][::fs,:].shape[0]-1
test_data : np.ndarray = np.zeros(shape=(length,3), dtype=float_dtype)
#!Current
test_data[:,0] = Data['Current(A)'][1:length+1]/3/6
#!Voltage and temperature of a Cell
test_data[:,1] = BMSsV[BMS_id][1:(length)*fs:fs,cell]
test_data[:,2] = BMSsT[BMS_id][1:(length)*fs:fs,cell]

normalised_test_data = np.divide(
    np.subtract(
            np.copy(a=test_data),
            MEAN
        ),
    STD
    )
PRED = np.zeros(shape=(length-500))
for i in trange(0, length-500):
    PRED[i] = model.predict(np.expand_dims(normalised_test_data[i:500+i,:], axis=0),
                batch_size=1)
time = np.linspace(0, BMSsV[BMS_id].shape[0]/fs/60, length-500)
plt.figure
plt.plot(time, PRED*100,
         label="Prediction", color='k', linewidth=7)
plt.grid(b=True, axis='both', linestyle='-', linewidth=1)
plt.ylabel('Charge(%)', fontsize=32)
#plt.xlabel('Samples fs=10', fontsize=32)
plt.xlabel('Time (minutes) - fs=10', fontsize=32)
plt.xticks(range(0,50,5), fontsize=18, rotation=0)
plt.yticks(range(10,90,5), fontsize=18, rotation=0)
plt.title(
      f"BMD ID-{BMS_id+1} {author}. {profile}-trained",
      fontsize=36)
plt.savefig(f'figures/BMD_ID-{BMS_id+1}-{author}-{profile}.png')
# %%