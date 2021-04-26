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
# """ sumN(0.5*(SoC-SoC*)^2) """
# chemali_loss = lambda y_true, y_pred: tf.keras.backend.sum(
#         x=tf.math.divide(
#                 x=tf.keras.backend.square(
#                         x=tf.math.subtract(
#                                 x=tf.cast(y_true, y_pred.dtype),
#                                 y=tf.convert_to_tensor(y_pred)
#                             )
#                     ),
#                 y=2
#             ),
#         axis=1
#     )
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
#? Model №1 - Chemali2017    - DST  - 1
#?                           - US06 - 50
#?                           - FUDS - 48
# author  : str = 'Chemali2017'
# iEpochs  : list = [1, 50, 48]

#? Model №2 - BinXiao2020    - DST  - 50
#?                           - US06 - 2 (21)
#?                           - FUDS - 48
# author  : str = 'BinXiao2020'
# iEpochs  : list = [50, 21, 48]

#? Model №3 - TadeleMamo2020 - DST  - 4
#?                           - US06 - 25
#?                           - FUDS - 10
# author  : str = 'TadeleMamo2020'
# iEpochs  : list = [None, 25, 10]

#? Model №7 - WeiZhang2020   - DST  - 9
#?                           - US06 - ?
#?                           - FUDS - 3
# author  : str = 'WeiZhang2020'
# iEpochs  : list = [9, None, 3]

# author  : str = 'Chemali2017'
# iEpochs  : list = [1 , 50, 48]

# author  : str = 'BinXiao2020'
# iEpochs  : list = [50, 21, 48]

author  : str = 'TadeleMamo2020'#'TadeleMamo2020'#'WeiZhang2020'#Chemali2017
iEpochs  : list = [4 , 25, 10]

models_loc : list = [f'../Models/{author}/DST-models/',
                     f'../Models/{author}/US06-models/',
                     f'../Models/{author}/FUDS-models/']
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
    models[i].compile(loss=tf.keras.losses.MeanAbsoluteError(),
            optimizer=tf.optimizers.Adam(learning_rate=0.001,
                    beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
            metrics=[tf.metrics.MeanAbsoluteError(),
                     tf.metrics.RootMeanSquaredError(),
                     tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
        )
# %%
#! Performing multipthreaded approach
devices : list = ['/cpu:0','/gpu:0','/gpu:1']
cpu_thread : threading.Thread = None
gpu_threads  : list = [threading.Thread]*2

use_cpu : bool = False
cpu_results : list = [None]
gpu_US_FUDS_results : list = [None]*2

def worker_evaluate(model, devise : str,
                    X:ndarray, Y:ndarray,
                    results : list, index : int
                    ) -> None:
    with tf.device(devise):
        results[index] = model.evaluate(x=X,y=Y,
                              batch_size=1, verbose=0,
                              sample_weight=None, steps=None, callbacks=None,
                              max_queue_size=10, workers=1,
                              use_multiprocessing=False, return_dict=False)
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
                             args=[models[1], devices[i+1],
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
                             args=[models[i+1], devices[i+1],
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
    print(f'results of DST trained model {profiles[0]} data:\n'
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
    print(f'results of DST trained model {profiles[0]} data:\n'
            f'\tMAE: {cpu_results[0][1]*100}%\n'
            f'\tRMSE:{cpu_results[0][2]*100}%\n'
            f'\tR2:  {cpu_results[0][3]*100}%\n')
# %%
