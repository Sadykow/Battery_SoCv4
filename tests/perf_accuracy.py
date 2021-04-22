#!/usr/bin/python
# %% [markdown]
# Using 3 devices run evaluation of the model
import threading

import tensorflow as tf  # Tensorflow and Numpy replacement
import tensorflow_addons as tfa

from numpy import ndarray

sys.path.append(os.getcwd() + '/..')
from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
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
    

# US06_window = WindowGenerator(Data=US06_Generator,
#                         input_width=500, label_width=1, shift=0,
#                         input_columns=['Current(A)', 'Voltage(V)',
#                                                 'Temperature (C)_1'],
#                         label_columns=['SoC(%)'], batch=1,
#                         includeTarget=False, normaliseLabal=False,
#                         shuffleTraining=False)
# FUDS_window = WindowGenerator(Data=FUDS_Generator,
#                         input_width=500, label_width=1, shift=0,
#                         input_columns=['Current(A)', 'Voltage(V)',
#                                                 'Temperature (C)_1'],
#                         label_columns=['SoC(%)'], batch=1,
#                         includeTarget=False, normaliseLabal=False,
#                         shuffleTraining=False)
# _, DSTx_train, DSTy_train = DST_window.train
# _, US06x_train, US06y_train = US06_window.train
# _, FUDSx_train, FUDSy_train = FUDS_window.train
# %%
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

#? Model №3 - TadeleMamo2020 - DST  - ?
#?                           - US06 - 25
#?                           - FUDS - 10
# author  : str = 'TadeleMamo2020'
# iEpochs  : list = [None, 25, 10]

#? Model №7 - WeiZhang2020   - DST  - 9
#?                           - US06 - ?
#?                           - FUDS - 3
# author  : str = 'WeiZhang2020'
# iEpochs  : list = [9, None, 3]

author  : str = 'BinXiao2020'#'TadeleMamo2020'#'WeiZhang2020'#Chemali2017
iEpochs  : list = [50, 21, 48]
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
                                # "AttentionWithContext": AttentionWithContext,
                                # "Addition": Addition,
                                }
                )
        )
    models[i].compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0005,
                beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Adamax'
                ),
            metrics=[tf.metrics.MeanAbsoluteError(),
                        tf.metrics.RootMeanSquaredError(),
                        tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
        )

# %%
#! Performing multipthreaded approach
devices : list = ['/cpu:0','/gpu:0','/gpu:1']
treads  : list = []
results : list = [None]*3

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

# Initialising threads
# for i in range(3):
#     treads.append(
#             threading.Thread(target=worker_evaluate,
#                              args=[models[0], devices[i],
#                                    X[i][:,:,:], Y[i][:,:],
#                                    results, i])
#         )
# GPUs only DST
for i in range(1, 3):
    treads.append(
            threading.Thread(target=worker_evaluate,
                             args=[models[0], devices[i],
                                   X[i][:,:,:], Y[i][:,:],
                                   results, i])
        )
    treads[i-1].start()
#! Use CPU thread as a daemon d.setDaemon(True)
#!https://www.bogotobogo.com/python/Multithread/python_multithreading_Daemon_join_method_threads.php
# Joining threads
for thread in treads:
    thread.join()
##
for i in range(1, len(results)):
    print(f'results of {profiles[i]} on model DST trained:\n'
          f'\tMAE: {results[i][1]*100}%\n'
          f'\tRMSE:{results[i][2]*100}%\n'
          f'\tR2:  {results[i][3]*100}%\n')
#%%
# GPU US06 based
treads  : list = []
results : list = [None]*3
for i in range(1, 3):
    treads.append(
            threading.Thread(target=worker_evaluate,
                             args=[models[1], devices[i],
                                   X[i][:,:,:], Y[i][:,:],
                                   results, i])
        )
    treads[i-1].start()

# Joining threads
for thread in treads:
    thread.join()

for i in range(1, len(results)):
    print(f'results of {profiles[i]} on model US06 trained:\n'
          f'\tMAE: {results[i][1]*100}%\n'
          f'\tRMSE:{results[i][2]*100}%\n'
          f'\tR2:  {results[i][3]*100}%\n')

# GPU FUDS based
treads  : list = []
results : list = [None]*3
for i in range(1, 3):
    treads.append(
            threading.Thread(target=worker_evaluate,
                             args=[models[2], devices[i],
                                   X[i][:,:,:], Y[i][:,:],
                                   results, i])
        )
    treads[i-1].start()

# Joining threads
for thread in treads:
    thread.join()

for i in range(1, len(results)):
    print(f'results of {profiles[i]} on model FUDS trained:\n'
          f'\tMAE: {results[i][1]*100}%\n'
          f'\tRMSE:{results[i][2]*100}%\n'
          f'\tR2:  {results[i][3]*100}%\n')

# GPU remaining
results : list = [None]*3
tr1 = threading.Thread(target=worker_evaluate,
                       args=[models[0], devices[1],
                            X[0][:,:,:], Y[0][:,:],
                            results, 0])
tr2 = threading.Thread(target=worker_evaluate,
                       args=[models[1], devices[2],
                            X[0][:,:,:], Y[0][:,:],
                            results, 1])
tr1.start()
tr2.start()
tr1.join()
tr2.join()

print(f'results of DST trained on DST:\n'
        f'\tMAE: {results[0][1]*100}%\n'
        f'\tRMSE:{results[0][2]*100}%\n'
        f'\tR2:  {results[0][3]*100}%\n')
print(f'results of US06 trained on DST:\n'
        f'\tMAE: {results[1][1]*100}%\n'
        f'\tRMSE:{results[1][2]*100}%\n'
        f'\tR2:  {results[1][3]*100}%\n')

# Complete the last remaining
worker_evaluate(models[2], devices[1],
                    X[0][:,:,:], Y[0][:,:],
                    results, 2)
print(f'results of FUDS trained on DST:\n'
        f'\tMAE: {results[2][1]*100}%\n'
        f'\tRMSE:{results[2][2]*100}%\n'
        f'\tR2:  {results[2][3]*100}%\n')
