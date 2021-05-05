# %%
import os, sys
from typing import get_type_hints
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.animation as animation

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
bms_file   : str  = 'SecondBalanceCharge.json'
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

# %%
#? Getting BMS data
BMSVoltages     : np.ndarray = np.empty(shape=(1,10), dtype=np.float32)
VoltageRT       : np.ndarray = np.empty(shape=(1,), dtype=np.float32)
BMSTemperatures : np.ndarray = np.empty(shape=(1,14), dtype=np.float32)
TemperatureRT   : np.ndarray = np.empty(shape=(1,), dtype=np.float32)
BMSsV = []
BMSsVt= []
BMSsT = []
BMSsTt= []
for i in range(8):
    BMSsV.append(BMSVoltages)
    BMSsVt.append(VoltageRT)
    BMSsT.append(BMSTemperatures)
    BMSsTt.append(TemperatureRT)
tic : int = perf_counter()
with open(bms_dir+bms_file) as file_object:
    # store file data in object
    lines = file_object.readlines()
c_lines = 0

for line in lines[-1::-1]:
    record = line.replace(':', ' : ').replace(',', '').replace('{','').replace('}','').replace('[','').replace(']','').split()
    #print(record)
    try:
        if(record[0] == '"VoltageInfo"'):
            BMSsV[int(record[7])] = np.append(
                                        arr=BMSsV[int(record[7])],
                                        values=np.array([float(v) for v in record[10:]], ndmin=2),
                                        axis=0)
            BMSsVt[int(record[7])] = np.append(
                                        arr=BMSsVt[int(record[7])],
                                        values=np.reshape(float(record[4]), newshape=(1,)),
                                        axis=0)
        elif(record[0] == '"TemperatureInfo"'):
            BMSsT[int(record[7])] = np.append(
                                        arr=BMSsT[int(record[7])],
                                        values=np.array([float(v) for v in record[10:]], ndmin=2),
                                        axis=0)
            BMSsTt[int(record[7])] = np.append(
                                        arr=BMSsTt[int(record[7])],
                                        values=np.reshape(float(record[4]), newshape=(1,)),
                                        axis=0)
        elif(record[0] == '"BalanceInfo"'):
            pass
        else:
            print("Unattended field: "+record[0])
        if(BMSsVt[0][1]-BMSsVt[0][-1] > 550):
            break
    except Exception as inst:
        print(f'Unusable record Line: {c_lines}')
    c_lines += 1
    # if(all(x.shape[0] >501 for x in BMSsV[:-1])):
    #     break
print(f'Parsing BMS data of {c_lines} lines took {perf_counter()-tic}')
for i in range(8):
    print(f"BMS:{i}: V:{BMSsV[i].shape} and T:{BMSsT[i].shape}")
    print(f"   :{i}: Vt:{BMSsVt[i].shape} and Tt:{BMSsTt[i].shape}")
# %%
# plt.figure()
# plt.plot(np.flipud(BMSsV[0][1:,0]))
# plt.title('Voltage')
# plt.figure()
# plt.plot(np.flipud(BMSsVt[0][1:]))
# plt.title('Time')
author  : str = 'Chemali2017'#'TadeleMamo2020'#'WeiZhang2020'#Chemali2017
profile : str = 'US06'#'FUDS'#'US06'#'DST'
iEpoch  : int = 50
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
# %%
fs = 4
BMS_id = 0
cell = 0 # 0-9=
length = 500
SoC : np.ndarray = np.zeros(shape=(10,), dtype=float_dtype)
charges = []
test_data : np.ndarray = np.zeros(shape=(length,3), dtype=float_dtype)
for BMS_id in range(0,7):
    for cell in range(0, 10):
        #!Current
        test_data[:,0] = 0.25
        #!Voltage and temperature of a Cell
        test_data[:,1] = BMSsV[BMS_id][:1:-fs,cell][:length]
        test_data[:,2] = np.repeat(BMSsT[BMS_id][:1:-fs,cell], 100)[:length]

        normalised_test_data = np.divide(
            np.subtract(
                    np.copy(a=test_data),
                    np.array([-0.35640615,  3.2060466 , 30.660755  ], dtype=float_dtype)
                ),
            np.array([ 0.9579658 ,  0.22374259, 13.653275  ], dtype=float_dtype)
            )
        SoC[cell] = model.predict(np.expand_dims(normalised_test_data, axis=0),
                        batch_size=1)[0][0]
    charges.append(np.copy(SoC))
# %%
# langs = ['4Cell-1', '4Cell-2', '4Cell-3', '4Cell-4', '4Cell-5', '4Cell-6', '4Cell-7', '4Cell-8', '4Cell-9', '4Cell-10']
# index = 1
# # Get a color map
# my_cmap = cm.get_cmap('jet_r')
# # Get normalize function (takes data in range [vmin, vmax] -> [0, 1])
# #my_norm = Normalize(vmin=0, vmax=8)
# plt.figure(num=None, figsize=(60, 36))
# for charge in charges[:]:
#     plt.subplot(2, 4, index)
#     plt.ylabel('SoC', fontsize=32)
#     plt.ylim([1,100])
#     plt.xticks(fontsize=24 )
#     plt.yticks(fontsize=32 )
#     plt.grid(b=True, axis='both', linestyle='-', linewidth=1)
#     plt.title(f'BMS: {index}', fontsize=32)
#     plt.bar(range(10),charge*100, color=my_cmap(charge))
#     index +=1
# plt.show()
# %%
# fig = plt.figure(num=None, figsize=(60,36))
# ax1 = fig.add_subplot(1,1,1)
my_cmap = cm.get_cmap('jet_r')

fig, axs = plt.subplots(2, 2)

#https://pythonprogramming.net/live-graphs-matplotlib-tutorial/
# def animate_old(i):
#     graph_data = open('example.txt','r').read()
#     lines = graph_data.split('\n')
#     xs = []
#     ys = []
#     for line in lines:
#         if len(line) > 1:
#             x, y = line.split(',')
#             xs.append(float(x))
#             ys.append(float(y))
#     ax1.clear()
#     ax1.plot(xs, ys)

def animate(i):
    for ax in axs.flat:
        # ax.clear()
        ax.bar(range(10),charges[i]*i, color=my_cmap(charges[i]))
    # for charge in charges[:1]:
        # plt.subplot(2, 4, index)
        # ax1.ylabel('SoC', fontsize=32)
        # ax1.ylim([1,100])
        # ax1.xticks(fontsize=24 )
        # ax1.yticks(fontsize=32 )
        # ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
        # ax1.title(f'BMS: {index}', fontsize=32)
        # ax1.bar(range(10),charge*i, color=my_cmap(charge))
        # index +=1

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()