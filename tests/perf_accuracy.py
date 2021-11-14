#!/usr/bin/python
# %% [markdown]
# Using 3 devices run evaluation of the model
import os, sys, threading

import tensorflow as tf  # Tensorflow and Numpy replacement
import tensorflow_addons as tfa

from numpy import ndarray

sys.path.append(os.getcwd() + '/..')
from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
from py_modules.Attention import *
# %%
physical_devices = tf.config.experimental.list_physical_devices()
print(f'Available devices: {physical_devices}')

for gpu in physical_devices[1:]:
    tf.config.experimental.set_memory_growth(
                                gpu, True)
# %%
Data    : str = '../Data/'
profiles: list = ['DST', 'US06', 'FUDS']
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
    
# %%
def chemali_loss(y_true, y_pred):
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
def jiao_loss(y_true : tf.Tensor, y_pred : tf.Tensor) -> tf.Tensor:
    y_pred = tf.convert_to_tensor(value=y_pred)
    y_true = tf.dtypes.cast(x=y_true, dtype=y_pred.dtype)        
    return (tf.math.squared_difference(x=y_pred, y=y_true))/2

#? Root Mean Squared Error loss function
zhang_loss = lambda y_true, y_pred: tf.sqrt(
            x=tf.reduce_mean(
                    input_tensor=tf.square(
                            x=tf.subtract(
                                x=tf.cast(x=y_true, dtype=y_pred.dtype),
                                y=tf.convert_to_tensor(value=y_pred)

                            )
                        ),
                    axis=0,
                    keepdims=False
                )
        )
#? Model №1 - Chemali2017    - DST  - 45
#?                           - US06 - 50
#?                           - FUDS - 48
# author  : str = 'Chemali2017'
# iEpochs  : list = [1, 50, 48]

#? Model №2 - BinXiao2020    - DST  - 50
#?                           - US06 - 50
#?                           - FUDS - 50
# author  : str = 'BinXiao2020'
# iEpochs  : list = [50, 21, 48]

#? Model №3 - TadeleMamo2020 - DST  - 19
#?                           - US06 - 25
#?                           - FUDS - 10
# author  : str = 'TadeleMamo2020'
# iEpochs  : list = [None, 25, 10]

#? Model №4 - MengJiao2020 -   DST  - 69
#?                         - d_US06 - 29
#?                         - d_FUDS - 68
# author  : str = 'MengJiao2020'
# iEpochs  : list = [69, 25, 10]

#? Model №5 - GelarehJavid2020 - DST  - 2
#?                             - US06 - 7 7
#?                             - FUDS - 7 8
author  : str = 'GelarehJavid2020'
iEpochs  : list = [10, 7, 8]

#? Model №6 - WeiZhang2020   - DST  - 9
#?                           - US06 - 3
#?                           - FUDS - 3
# author  : str = 'WeiZhang2020'
# iEpochs  : list = [9, 3, 3]

models_loc : list = [f'../Models/{author}/DST-models/',
                     f'../Models/{author}/US06-models/',
                     f'../Models/{author}/FUDS-models/']
#?! Model №4 - MengJiao2020 ONLY
# models_loc : list = [f'../Models/{author}/DST-models/',
#                      f'../Models/{author}/d_US06-models/',
#                      f'../Models/{author}/d_FUDS-models/']
models = []
for i in range(3):
    models.append(
            tf.keras.models.load_model(
                f'{models_loc[i]}{iEpochs[i]}',
                compile=False,
                custom_objects={"RSquare": tfa.metrics.RSquare,
                                "AttentionWithContext": AttentionWithContext,
                                "Addition": Addition,
                                }
                )
        )
    #! 1) Chemali compile
    # models[i].compile(loss=chemali_loss,
    #         optimizer=tf.optimizers.Adam(learning_rate=0.001,
    #                 beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
    #         metrics=[tf.metrics.MeanAbsoluteError(),
    #                     tf.metrics.RootMeanSquaredError(),
    #                     tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
    #     )
    #! 2) Xiao
    # models[i].compile(
    #         loss=tf.keras.losses.MeanSquaredError(),
    #         optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0005,
    #             beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Adamax'
    #             ),
    #         metrics=[tf.metrics.MeanAbsoluteError(),
    #                     tf.metrics.RootMeanSquaredError(),
    #                     tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
    #     )
    #! 3) Mamo
    # models[i].compile(loss=tf.keras.losses.MeanAbsoluteError(),
    #         optimizer=tf.optimizers.Adam(learning_rate=0.001,
    #                 beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
    #         metrics=[tf.metrics.MeanAbsoluteError(),
    #                  tf.metrics.RootMeanSquaredError(),
    #                  tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
    #     )
    #! 4) Jiao
    # models[i].compile(loss=jiao_loss,
    #          optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,
    #                 momentum=0.3, nesterov=False, name='SGDwM'),
    #           metrics=[tf.metrics.MeanAbsoluteError(),
    #                    tf.metrics.RootMeanSquaredError(),
    #                    tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
    #         )
    #! 5) Javid
    # models[i].compile(loss=tf.keras.losses.MeanAbsoluteError(),
    #         optimizer=tf.optimizers.Adam(learning_rate=0.001,
    #                 beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
    #         metrics=[tf.metrics.MeanAbsoluteError(),
    #                     tf.metrics.RootMeanSquaredError(),
    #                     tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
    #     )
    #! 5) Zhang
    models[i].compile(loss=zhang_loss,
        optimizer=tf.optimizers.Adam(learning_rate=0.0001,
                beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
        metrics=[tf.metrics.MeanAbsoluteError(),
                    tf.metrics.RootMeanSquaredError(),
                    tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
        )
# %%
#! Performing multipthreaded approach
devices : list = ['/cpu:0','/gpu:0','/gpu:1']
r_devices : list = ['/cpu:0','/gpu:1','/gpu:0']
cpu_thread : threading.Thread = None
gpu_threads  : list = [threading.Thread]*2

use_cpu : bool = True
cpu_results : list = [None]
gpu_US_FUDS_results : list = [None]*2

def worker_evaluate(model, devise : str,
                    X:ndarray, Y:ndarray,
                    results : list, index : int
                    ) -> None:
    with tf.device(devise):
        results[index] = model.evaluate(x=X,y=Y,
                              batch_size=1, verbose=1,
                              sample_weight=None, steps=None, callbacks=None,
                              max_queue_size=10, workers=1,
                              use_multiprocessing=True, return_dict=False)
#? models[0] -- DST
#? models[1] -- US06
#? models[2] -- FUDS
if use_cpu: # Use CPU?: models[0] -->> DST : ignore do on GPU
    cpu_tread = threading.Thread(target=worker_evaluate,
                                args=[models[0], devices[0],
                                    X[0][:,:,:], Y[0][:,:],
                                    cpu_results, 0])
# GPU DST model based
for i in range(0, len(gpu_threads)):
    gpu_threads[i] = threading.Thread(target=worker_evaluate,
                             args=[models[0], devices[i+1],
                                   X[i+1][:,:,:], Y[i+1][:,:],
                                   gpu_US_FUDS_results, i])
    gpu_threads[i].start()
        
#! Use CPU thread as a daemon d.setDaemon(True)
#!https://www.bogotobogo.com/python/Multithread/python_multithreading_Daemon_join_method_threads.php
# Joining threads
if use_cpu:
    cpu_tread.setDaemon(True)
    cpu_tread.start()

for thread in gpu_threads:
    thread.join()

for i in range(0, len(gpu_US_FUDS_results)):
    print(f'1st stage - results of DST trained model {profiles[i+1]} data:\n'
          f'\tMAE: {gpu_US_FUDS_results[i][1]*100}%\n'
          f'\tRMSE:{gpu_US_FUDS_results[i][2]*100}%\n'
          f'\tR2:  {gpu_US_FUDS_results[i][3]*100}%\n')

# %% GPU US06 model based
for i in range(0, len(gpu_threads)):
    gpu_threads[i] = threading.Thread(target=worker_evaluate,
                             args=[models[1], r_devices[i+1],
                                   X[i+1][:,:,:], Y[i+1][:,:],
                                   gpu_US_FUDS_results, i])
    gpu_threads[i].start()

# Joining threads
for thread in gpu_threads:
    thread.join()

for i in range(0, len(gpu_US_FUDS_results)):
    print(f'2nd stage - results of US06 trained model {profiles[i+1]} data:\n'
          f'\tMAE: {gpu_US_FUDS_results[i][1]*100}%\n'
          f'\tRMSE:{gpu_US_FUDS_results[i][2]*100}%\n'
          f'\tR2:  {gpu_US_FUDS_results[i][3]*100}%\n')

# %% GPU FUDS model based
for i in range(0, len(gpu_threads)):
    gpu_threads[i] = threading.Thread(target=worker_evaluate,
                             args=[models[2], devices[i+1],
                                   X[i+1][:,:,:], Y[i+1][:,:],
                                   gpu_US_FUDS_results, i])
    gpu_threads[i].start()

# Joining threads
for thread in gpu_threads:
    thread.join()

for i in range(0, len(gpu_US_FUDS_results)):
    print(f'3rd stage - results of FUDS trained model {profiles[i+1]} data:\n'
          f'\tMAE: {gpu_US_FUDS_results[i][1]*100}%\n'
          f'\tRMSE:{gpu_US_FUDS_results[i][2]*100}%\n'
          f'\tR2:  {gpu_US_FUDS_results[i][3]*100}%\n')

# %% GPU remaining US06 - DST and FUDS - DST
for i in range(0, len(gpu_threads)):
    gpu_threads[i] = threading.Thread(target=worker_evaluate,
                             args=[models[i+1], r_devices[i+1],
                                   X[0][:,:,:], Y[0][:,:],
                                   gpu_US_FUDS_results, i])
    gpu_threads[i].start()

for thread in gpu_threads:
    thread.join()

for i in range(0, len(gpu_US_FUDS_results)):
    print(f'4th stage - results of {profiles[i+1]} trained model DST data:\n'
          f'\tMAE: {gpu_US_FUDS_results[i][1]*100}%\n'
          f'\tRMSE:{gpu_US_FUDS_results[i][2]*100}%\n'
          f'\tR2:  {gpu_US_FUDS_results[i][3]*100}%\n')

# %% Complete the last remaining
if use_cpu:
    while(cpu_tread.is_alive()):
        pass
    print(f'5th stage - results of DST trained model {profiles[0]} data:\n'
            f'\tMAE: {cpu_results[0][1]*100}%\n'
            f'\tRMSE:{cpu_results[0][2]*100}%\n'
            f'\tR2:  {cpu_results[0][3]*100}%\n')
else: #! Complete on GPU last part
    cpu_tread = threading.Thread(target=worker_evaluate,
                                args=[models[0], devices[1],
                                    X[0][:,:,:], Y[0][:,:],
                                    cpu_results, 0])
    cpu_tread.start()
    cpu_tread.join()
    print(f'5th stage - results of DST trained model {profiles[0]} data:\n'
            f'\tMAE: {cpu_results[0][1]*100}%\n'
            f'\tRMSE:{cpu_results[0][2]*100}%\n'
            f'\tR2:  {cpu_results[0][3]*100}%\n')
# %%
