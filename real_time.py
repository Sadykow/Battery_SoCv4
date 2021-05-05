# %%
#import pandas as pd
import os, sys
import numpy as np
import pandas as pd

import socket
import time
from Data.backend.can_parser import raw_can_msg, split_can_msg
from Data.backend.can_ids import BMS_TransmitVoltage_ID, BMS_TransmitTemperature_ID

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
import tensorflow_addons as tfa

from datetime import datetime
from time import perf_counter
from tqdm import trange

from extractor.DataGenerator import *
from py_modules.Attention import *

float_dtype : type = np.float32
int_dtype   : type = np.int32

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
log = []
# Setup TCP Connection for CAN1
TCP_IP = '192.168.0.7'
TCP_PORT_CAN2 = 20005 # Double check this
BUFFER_SIZE = 4096
ID_TYPE = 1

start = (time.time_ns() // 1_000_000 )

file_open = open("Data/BMS_data/Log_" + str(start) + ".txt", 'a')
file_raw = open("Data/BMS_data/rLog_" + str(start) + ".txt", 'w+b')
file_open.write(str(start))

#   Connect to the TCP server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT_CAN2))

# s.send(b'\x84\x10\x50\x80\x02\x01\x02\x04\x08\x00\x00\x00\x00')
# s.send(b'\x84\x00\x00\x06\x78\x12\x34\x56\x78\x00\x00\x00\x00')
# Receive any data that is available for us
data = s.recv(BUFFER_SIZE)
file_raw.write(data)
# print(len(data))
# print(data)
s.close()
# with open('Data/BMS_data/rLog_1620195401961.txt', 'rb') as file:
#     data = file.read()
raw_msgs = []
# %%
current_ms = (time.time_ns() // 1_000_000 ) - start
for idx in range(0, len(data)-1):
    # print(f"len: {len(data)} idx: {idx} remaining: {len(data)-idx}")      
    ethernetPacketInformation = data[idx]
    dataLength = (ethernetPacketInformation & 0xF)
    # CAN ID
    idx= idx+1
    canId = (data[idx] << 24 | data[idx+1] << 16 | data[idx+2] << 8 | data[idx+3] << 0)
    idx = idx + 4
    parsedData = data[idx:dataLength+idx]
    raw_msgs.append(raw_can_msg(current_ms, canId, ID_TYPE, dataLength, parsedData))
    file_open.write(str(raw_can_msg(current_ms, canId, ID_TYPE, dataLength, parsedData)) + "\n")
    idx = idx + 8
# %%
# BMS MESSAGES (Battery Management System)

def parse_bms_transmit_voltage(msg: raw_can_msg):
    bmsID = msg.id & 0xF
    msgID = (msg.data[0] >> 6) & 0x3
    voltages = []
    for i in range(0, int(msg.data_length / 2)):
        v_h = (int(msg.data[2 * i + 1] & 0x3F)) << 6
        v_l = int(msg.data[2 * i]) & 0x3F
        voltage = v_h | v_l
        voltages.append(voltage)
    
    str_voltages = [str(int) for int in voltages]
    str_voltages = ", ".join(str_voltages)

    return split_can_msg(
            msg.timestamp, "BMS_TransmitVoltage", ["ID: ", bmsID, "MSG_ID: " , msgID, " VOLT: ", str_voltages]
        ).to_array()

def parse_bms_transmit_temperature(msg: raw_can_msg):
    bmsID = msg.id & 0xF
    msgID = msg.data[0] & 0x1
    temperatures = []
    for i in range(1, msg.data_length - 1):
        temperatures.append(msg.data[i])

    str_temps = [str(int) for int in temperatures]
    str_temps = ", ".join(str_temps)

    return split_can_msg(
            msg.timestamp, "BMS_TransmitTemperature", ["BMSID: ", bmsID," MSGID: ", msgID," TEMPS: ", str_temps]
        ).to_array()

# result = parse_can_msgs(raw_msgs, False)
# output = ""

# for msg in result:
#     print(str(msg))
# %%
#? Getting BMS data
BMSVoltages     : np.ndarray = np.empty(shape=(1,4), dtype=np.float32)
BMSTemperatures : np.ndarray = np.empty(shape=(1,5), dtype=np.float32)
BMSsV = []
BMSsT = []
for i in range(6):
    BMSsV.append(BMSVoltages)
    BMSsT.append(BMSTemperatures)
BMSsVid = [BMSsV]*3
BMSsTid = [BMSsT]*3

msg = raw_msgs[14]
c_lines = 0
tic : int = perf_counter()
for msg in raw_msgs[:]:
    parsed = None
    id_no_bmsid = msg.id & (~0b1111)  # Ignore last 4 bits
    if id_no_bmsid == BMS_TransmitVoltage_ID:
        parsed = parse_bms_transmit_voltage(msg)[2]
        #       MSG_ID      BMS_ID
        chunk = parsed[-1][:].replace(',','').split()
        values = np.array([float(v)/1000 for v in chunk], ndmin=1)
        BMSsVid[parsed[3]][parsed[1]] = np.append(
                                    arr=BMSsVid[parsed[3]][parsed[1]],
                                    values=np.expand_dims(values, axis=0),
                                    axis=0)
    elif id_no_bmsid == BMS_TransmitTemperature_ID:
        parsed = parse_bms_transmit_temperature(msg)[2]
        #       MSG_ID      BMS_ID
        chunk = parsed[-1][:].replace(',','').split()
        values = np.array([int(v) for v in chunk], ndmin=1)
        BMSsTid[parsed[3]][parsed[1]] = np.append(
                                    arr=BMSsTid[parsed[3]][parsed[1]],
                                    values=np.expand_dims(values, axis=0),
                                    axis=0)
    c_lines += 1
print(f'Parsing BMS data of {c_lines} lines took {perf_counter()-tic}')
# %%
# for msg in raw_msgs[:500]:
#     parsed = None
#     id_no_bmsid = msg.id & (~0b1111)  # Ignore last 4 bits
#     if id_no_bmsid == BMS_TransmitVoltage_ID:
#         parsed = parse_bms_transmit_voltage(msg)[2]
#         print(parsed)
#     elif id_no_bmsid == BMS_TransmitTemperature_ID:
#         parsed = parse_bms_transmit_temperature(msg)[2]
#         print(parsed)
BMSVoltages     : np.ndarray = np.empty(shape=(1,10), dtype=np.float32)
for id in range(6):
    for msg in len(1, BMSsVid[0][0]):
        np.concatenate((BMSsVid[0][id][i], BMSsVid[1][id][i], BMSsVid[2][id][i]))
BMSTemperatures : np.ndarray = np.empty(shape=(1,14), dtype=np.float32)

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
author  : str = 'Chemali2017'#'TadeleMamo2020'#'WeiZhang2020'#Chemali2017
profile : str = 'US06'#'FUDS'#'US06'#'DST'
iEpoch  : int = 50
model_loc : str = f'../Models/{author}/{profile}-models/'

try:
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