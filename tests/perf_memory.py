#!/usr/bin/python
# %% [markdown]
# # conda activate cTF2.4.1 to get Memory usage
# # conda activate TF2.4-CPU to get Average time usage
# %%
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

sys.path.append(os.getcwd() + '/..')
from py_modules.Attention import *
# %%
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    #! With /device/GPU:1 the output was faster.
    #! need to research more why.
    tf.config.experimental.set_visible_devices(
                            physical_devices[1], 'GPU')

    #if GPU == 1:
    tf.config.experimental.set_memory_growth(
                            physical_devices[1], True)
    logging.info("GPU found and memory growth enabled") 
    
    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    logging.info("GPU found") 
    logging.debug(f"\nPhysical GPUs: {len(physical_devices)}"
                  f"\nLogical GPUs: {len(logical_devices)}")
#! For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float32')
# %%
profile : str = 'FUDS'
file_name : str = 'Chemali2017'
model_loc : str = f'../Models/{file_name}/{profile}-models/'
iEpoch = 0
firstLog  : bool = True
try:
    for _, _, files in os.walk(model_loc):
        for file in files:
            if file.endswith('.ch'):
                iEpoch = int(os.path.splitext(file)[0])
    
    lstm_model : tf.keras.models.Sequential = tf.keras.models.load_model(
            f'{model_loc}{iEpoch}',
            compile=False,
            custom_objects={"RSquare": tfa.metrics.RSquare,
                            "AttentionWithContext": AttentionWithContext,
                            "Addition": Addition,
                            })
    firstLog = False
    print("Model Identefied. Continue training.")
except OSError as identifier:
    print("Model Not Found, creating new. {} \n".format(identifier))

log_dir=model_loc+ \
    f'tensorboard/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1, write_graph=True, write_images=False,
        update_freq='epoch', profile_batch=(4,8), embeddings_freq=0,
        embeddings_metadata=None
    )
lstm_model.compile(loss=tf.losses.MeanAbsoluteError(),
        optimizer=tf.optimizers.Adam(learning_rate=0.001,
                beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
        metrics=[tf.metrics.MeanAbsoluteError(),
                    tf.metrics.RootMeanSquaredError(),
                    tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
    )
# %%
x_train = np.zeros(shape=(10,500,3), dtype=np.float32)
y_train = np.zeros(shape=(10,1), dtype=np.float32)

# history = lstm_model.fit(x=x_train, y=y_train, epochs=1,
#                     callbacks=[tensorboard_callback],
#                     batch_size=1, shuffle=True
#                     )
# %%
optimiser = tf.optimizers.Adam()
loss_fn   = tf.losses.MeanAbsoluteError()
def train_single_st(x, y, prev_loss):
    with tf.GradientTape() as tape:
        logits     = lstm_model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, lstm_model.trainable_weights)
    # optimiser.update_loss(prev_loss, loss_value)
    optimiser.apply_gradients(zip(grads, lstm_model.trainable_weights))
    return loss_value
loss_value : np.float32 = 1.0
tf.profiler.experimental.start(log_dir)
for step in trange(10):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        # logit = test_step(x_train[step:step+1,:,:])
        # loss_value = loss_fn(y_train[step:step+1], logit)
        loss_value = train_single_st(x_train[step:step+1,:,:],
                                     y_train[step:step+1,:],
                                     loss_value)
tf.profiler.experimental.stop()
# %%