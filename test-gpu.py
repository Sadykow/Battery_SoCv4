#!/usr/bin/python
# %% [markdown]
# # # 2
# # #
# # LSTM with Attention Mechanism for SoC by Tadele Mamo - 2020
# 

# Data windowing has been used as per: Ψ ={X,Y} where:
# %%
import logging
import os, sys, getopt  # OS, SYS, argc functions
from sys import platform  # Get type of OS

import tensorflow as tf

from extractor.utils import str2bool
# %%
# Initialise the logger
try:
    opts, args = getopt.getopt(sys.argv[1:],"hd:e:g:p:",
                    ["help", "debug=", "epochs=",
                     "gpu=", "profile="])
except getopt.error as err: 
    # output error, and return with an error code 
    print (str(err)) 
    print ('EXEPTION: Arguments requied!')
    sys.exit(2)

# opts = [('-d', 'False'), ('-e', '50'), ('-g', '1'), ('-p', 'DST')]

for opt, arg in opts:
    if opt == '-h':
        print('HELP: Use following default example.')
        print('test.py --dir=../Data --out=SavedModel/LSTM')
        print('TODO: Create a proper help')
        sys.exit()
    elif opt in ("-d", "--debug"):
        if(str2bool(arg)):
            logging.warning("Logger enabled")
            logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s --> %(levelname)s:%(message)s')
        else:
            logging.basicConfig(level=logging.CRITICAL)
    elif opt in ("-e", "--epochs"):
        epochs = int(arg)
    elif opt in ("-g", "--gpu"):
        gpu_id = int(arg)
    elif opt in ("-p", "--profile"):
        profile = (arg)

#! Select GPU for usage. CPU versions ignores it
GPU=gpu_id
physical_devices = tf.config.list_physical_devices('GPU')
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
print(physical_devices)

print(f'Variable Epochs: {epochs} of type: {type(epochs)}')
print(f'Variable GPU: {gpu_id} of type: {type(gpu_id)}')
print(f'Variable Progile: {profile} of type: {type(profile)}')
