#!/home/sybil/miniconda3/envs/TF2.8/bin/python
# %% [markdown]
# Using combination of 3 devices, along with Clickhouse database to compute 150
# models from clickhouse and store them back at individual records
from operator import mod
import os, sys, threading
from tabnanny import verbose

import tensorflow as tf  # Tensorflow and Numpy replacement
import tensorflow_addons as tfa

from numpy import ndarray

from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
from py_modules.Attention import *

from clickhouse_driver import Client
from tqdm import trange
from typing import Callable

import gc

from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient
from getpass import getpass

from itertools import chain
#   996  conda install --name TF2.8 ptyprocess --update-deps --force-reinstall
#   997  conda install --name TF2.8 wcwidth --update-deps --force-reinstall
# %%
physical_devices = tf.config.experimental.list_physical_devices()
print(f'Available devices:\n{physical_devices}')

for gpu in physical_devices[1:]:
    tf.config.experimental.set_memory_growth(
                                gpu, False)
# %%
#! Clickhouse establesment
database = 'ml_base'
models : str = 'models'
mapper : str = 'mapper'
accuracies : str = 'accuracies'
client = Client(host='192.168.1.254', port=9000, 
                database=database, user='tf', password='TF28')
# %%
# client.execute('SHOW DATABASES')

# columns = (f'CREATE TABLE IF NOT EXISTS {database}.{accuracies} ('
# #                            "id UInt64, "
#                             "File UInt64, "
#                             'Attempt UInt32, '
#                             'Profile String, '
#                             # 'mae Float32, '
#                             # 'rms Float32, '
#                             # 'r_s Float32 '
#                             'logits Array(Float32)'
#                         ') '
#                         'ENGINE = MergeTree() '
#                         'ORDER BY (File, Attempt, Profile);'
#                 )
# client.execute(columns)
def createSSHClient(server : str, port : int,
                    user : str, password : str) -> SSHClient:
    client : SSHClient = SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.connect(hostname = server, port = port,
                   username = user, password = password)
    return client

def progress4(filename, size, sent, peername) -> None:
    sys.stdout.write(
        "(%s:%s) %s's progress: %.2f%%   \r" % (peername[0], peername[1],
                                    filename, float(sent)/float(size)*100)
        )

username : str = 'n9312706' # getpass(prompt='QUT ID: ', stream=None)

ssh = createSSHClient(
        server='lyra.qut.edu.au',
        port=22,
        user=username,
        password=getpass(prompt='Password: ', stream=None)
    )
remote_path : str = f'/mnt/home/{username}/MPhil/TF/Battery_SoCv4/Mods/'

scp = SCPClient(ssh.get_transport(), progress4=progress4)
# %%
Data    : str = 'Data/'
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
    x, y = WindowGenerator(Data=g,
                    input_width=500, label_width=1, shift=0,
                    input_columns=['Current(A)', 'Voltage(V)',
                                            'Temperature (C)_1'],
                    label_columns=['SoC(%)'], batch=1,
                    includeTarget=False, normaliseLabal=False,
                    shuffleTraining=False,
                    round=5).train
    X.append(x)
    Y.append(y)
#! del dataGenerators
# %%
#! Get the model from Clickhouse
def get_models(files : list[int]) -> list:
    tmp_folder = 'Modds/tmp/'
    modds : list = []
    #* 2) Extract models to a list
    for c in files:
        # print(f'SELECT latest FROM {database}.{models} WHERE File = {c}')
        with open(f'{tmp_folder}model', 'wb') as file:
            file.write(
                client.execute(
                        f'SELECT latest FROM {database}.{models} WHERE File = {c}'
                    )[0][0]
                )
        modds.append(
                tf.keras.models.load_model(f'{tmp_folder}model', compile=False)
            )
        os.remove(f'{tmp_folder}model')
    return modds

@tf.function
def test_step(model, input : tuple[np.ndarray, np.ndarray]) -> tf.Tensor:
    return model(input, training=False)

def valid_loop(model, dist_input  : tuple[np.ndarray, np.ndarray],
               verbose : int = 0) -> tf.Tensor:
    x, y = dist_input
    logits  : tf.Variable = tf.Variable(0.0)
    val_MAE     = tf.metrics.MeanAbsoluteError()
    val_RMSE    = tf.metrics.RootMeanSquaredError()
    val_RSquare = tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)
    
    # Debug verbose param
    if verbose == 1:
        rangeFunc : Callable = trange
    else:
        rangeFunc : Callable = range
    
    #! Prediction on this part can be paralylised across two GPUs. Like OpenMP
    for i in rangeFunc(y.shape[0]):
        logits = test_step(model, x[i,:,:,:])
        # logits = model(x[i,:,:,:], training=False)
        val_MAE.update_state(y_true=y[i],     y_pred=logits)
        val_RMSE.update_state(y_true=y[i],    y_pred=logits)
        val_RSquare.update_state(y_true=y[i], y_pred=logits)
        
        # mae[i] = val_MAE.result()
        # rmse[i] = val_RMSE.result()
        # rsquare[i] = val_RSquare.result()
    #! Error with RMSE here. No mean should be used.
    # return [loss, mae, rmse, rsquare, logits]
    mae      : float = val_MAE.result()
    rmse     : float = val_RMSE.result()
    r_Square : float = val_RSquare.result()
    
    # Reset training metrics at the end of each epoch
    val_MAE.reset_states()
    val_RMSE.reset_states()
    val_RSquare.reset_states()

    return [mae, rmse, r_Square]

def worker_evaluate(model, device : str,
                    X:ndarray, Y:ndarray,
                    file : int, attempt : int, profile : str,
                    ) -> None:
    tic : float = time.perf_counter()
    model.compile(
        loss=tf.losses.MeanAbsoluteError(
                    reduction=tf.keras.losses.Reduction.NONE),
        optimizer=tf.optimizers.Adam(learning_rate=0.0001,
                beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
        metrics=[tf.metrics.MeanAbsoluteError(),
                 tf.metrics.RootMeanSquaredError(),
                 tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
        )
    #* Compute on a Device
    with tf.device(device):
    #     metrics = valid_loop(model=model, dist_input=(X,Y), verbose=0)
        # metrics = model.evaluate(x=X[:,0,:,:],y=Y[:,0],
        #             batch_size=1, verbose=0,
        #             sample_weight=None, steps=None, callbacks=None,
        #             max_queue_size=10, workers=1,
        #             use_multiprocessing=True, return_dict=False)[1:]
        metrics = model.predict(x=X[:,0,:,:],
                    batch_size=1, verbose=0,
                    steps=None, callbacks=None,
                    max_queue_size=10, workers=1,
                    use_multiprocessing=True)

    toc : float = time.perf_counter() - tic
    #* Send Metrics along with details
    # client.execute(
    #         f'INSERT INTO {database}.{accuracies} '
    #         '(File, Attempt, Profile, mae, rms, r_s) VALUES ',
    #         [{'File' : file, 'Attempt' : attempt, 'Profile' : profile,
    #         'mae' : metrics[0], 'rms' : metrics[1], 'r_s' : metrics[2]}]
    #     )
    client.execute(
            f'INSERT INTO {database}.{accuracies} '
            '(File, Attempt, Profile, logits) VALUES ',
            [{'File' : file, 'Attempt' : attempt, 'Profile' : profile,
            'logits' : metrics}]
        )
    print(f'>{device} Worker: {file} on {attempt}x{profile} is over in {toc}!',
          flush=True)


#! Distributed strategy
def worker(model, devise : str,
                    X:ndarray, Y:ndarray,
                    metrics : list, index : int
                    ) -> None:
    model.compile(
        loss=tf.losses.MeanAbsoluteError(
                    reduction=tf.keras.losses.Reduction.NONE),
        optimizer=tf.optimizers.Adam(learning_rate=0.0001,
                beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
        metrics=[tf.metrics.MeanAbsoluteError(),
                 tf.metrics.RootMeanSquaredError(),
                 tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
        )
    #! Custom loop makes GPU utilise only 40% for both
    #! whereas compiled one foces them to work separately 80/52
    with tf.device(devise):
    #     metrics[index] = func(model=model, dist_input=(X,Y), verbose=0)
        # metrics[index] = model.evaluate(x=X[:,0,:,:],y=Y[:,0],
        #             batch_size=1, verbose=0,
        #             sample_weight=None, steps=None, callbacks=None,
        #             max_queue_size=10, workers=1,
        #             use_multiprocessing=True, return_dict=False)[1:]
        metrics[index] = model.predict(x=X[:,0,:,:],
                    batch_size=1, verbose=0,
                    steps=None, callbacks=None,
                    max_queue_size=10, workers=1,
                    use_multiprocessing=True)
    # del X, Y

def distributed_strategy(model, devices : str,
                        X:ndarray, Y:ndarray,
                        file : int, attempt : int, profile : str,
                        ) -> None:
    gpu0 : threading.Thread = None
    gpu1 : threading.Thread = None
    metrics : list = [None]*2
    split = int(X.shape[0]*0.61)
    #? Shared model only uses 40% utilisation on both GPUs, where as 80% and 50%
    # dublicated = tf.keras.models.clone_model(model)

    tic : float = time.perf_counter()
    #? GPU at 40%. Try Copy or DeepCopy of the inputs
    gpu0 = threading.Thread(target=worker,
                            args=[model, devices[0],
                            X[:split,:,:], Y[:split,:],
                            metrics, 0])
    gpu1 = threading.Thread(target=worker,
                            args=[model, devices[1],
                            X[split:,:,:], Y[split:,:],
                            metrics, 1])
    gpu0.start()
    gpu1.start()
    gpu0.join()
    gpu1.join()

    toc : float = time.perf_counter() - tic
    averages = []
    for first, second in zip(metrics[0], metrics[1]):
        # print( ((first+second)/2), sep=', ')
        averages.append( ((first+second)/2) )
    
    #* Send Metrics along with details
    print(metrics)
    print(averages)
    client.execute(
            f'INSERT INTO {database}.{accuracies} '
            '(File, Attempt, Profile, mae, rms, r_s) VALUES ',
            [{'File' : file, 'Attempt' : attempt, 'Profile' : profile,
            'mae' : averages[0], 'rms' : averages[1], 'r_s' : averages[2]}]
        )
    print(f'>Double Worker: {file} on {attempt}x{profile} is over in {toc}!',
          flush=True)
    # del dublicated

def multi_cpu(model, device : str, n_threads : int,
                    X:ndarray, Y:ndarray,
                    file : int, attempt : int, profile : str,
                    ) -> None:
    threads : list = [threading.Thread]*n_threads
    metrics : list = [None]*n_threads
    split = int(X.shape[0]/n_threads)
    tic : float = time.perf_counter()
    for i in range(0, n_threads):
        threads[i] = threading.Thread(target=worker,
                            args=[model, device,
                            X[i*split:(i+1)*split,:,:], Y[i*split:(i+1)*split,:],
                            metrics, i])
        threads[i].start()
    for thread in threads:
        thread.join()
    toc : float = time.perf_counter() - tic
    # print(metrics)
    # averages = [0]*3
    # for j in range(0, len(averages)):
    #     for i in range(0, n_threads):
    #         averages[j] += metrics[i][j]
    #     averages[j] = averages[j]/n_threads
    
    #* Send Metrics along with details
    # client.execute(
    #         f'INSERT INTO {database}.{accuracies} '
    #         '(File, Attempt, Profile, mae, rms, r_s) VALUES ',
    #         [{'File' : file, 'Attempt' : attempt, 'Profile' : profile,
    #         'mae' : averages[0], 'rms' : averages[1], 'r_s' : averages[2]}]
    #     )
    client.execute(
            f'INSERT INTO {database}.{accuracies} '
            '(File, Attempt, Profile, logits) VALUES ',
            [{'File' : file, 'Attempt' : attempt, 'Profile' : profile,
            'logits' : list(chain.from_iterable(metrics))}]
        )
    print(f'>{device} Worker: {file} on {attempt}x{profile} is over in {toc}!',
          flush=True)

# %%
def get_meta(c_file) -> tuple:
    return client.execute(
f"""
SELECT
    ModelID,
    Name,
    Attempt,
    Profile
FROM ml_base.mapper
WHERE File = {c_file}
"""
    )[0]
def get_epoch(c_file) -> int:
    return client.execute(
f"""
SELECT MIN(Epoch)
FROM ml_base.faulties
WHERE File = {c_file}
"""
    )[0][0]
# %%
#TODO: Sybil:20cores,2GPU: 3h , 10:18 mins, 10:47 mins
#TODO: Compiled: 1t:22:33min , 8:20 mins, 10:47 mins
#TODO: Compiled: 2t:11:30min , (5.8)mins
#TODO: Compiled: 3t:8:11min , (5.8)mins
#TODO: Compiled: 4t:6:35min , (5.8)mins
#TODO: Compiled: 5t:5:20min , (5.8)mins
devices : list = ['/cpu:0','/gpu:0','/gpu:1']
r_devices : list = ['/cpu:0','/gpu:1','/gpu:0']
cpu_thread : threading.Thread = None
gpu_threads  : list = [threading.Thread]*2

#! Get list of Files belonging to Chemali2017 and attampt 1
#! For every extracted file do:
author  : str = 'Chemali2017'
id_author : int = 1
attempts : range = range(1,11)
# a = 4
# for a in attempts:
tmp_folder = 'Modds/tmp/'
for c in range(1,31):
    #* 1) Locate how many files stored
    # c_files = client.execute(
    #     f"SELECT File FROM {database}.{mapper} WHERE Name = '{author}' AND Attempt = '{a}'"
    # )
    # c_files = [c[0] for c in c_files]
    
    #* 2) Extract models
    print('Extracting models')
    # modds = get_models(files=c_files)
    # with open(f'{tmp_folder}model', 'wb') as file:
    #     file.write(
    #         client.execute(
    #                 f'SELECT testi FROM {database}.{models} WHERE File = {c}'
    #             )[0][0]
    #         )
    meta = get_meta(c_file=c)
    folder_path = f'ModelsUp-{id_author}/3x{meta[1]}-(131)/{meta[2]}-{meta[3]}/'
    epoch = get_epoch(c_file=c)
    epoch -= 1
    path = remote_path + folder_path + str(epoch)
    scp.get(path, f'{tmp_folder}model')

    model = tf.keras.models.load_model(f'{tmp_folder}model', compile=False)
    a = client.execute(
                    f'SELECT Attempt FROM {database}.{mapper} WHERE File = {c}'
                )[0][0]
    #* 3) Set up the order
    #* DST-60066, US06-56661, FUDS-58613
    #* CPU-US06 , GPU0-DST, GPU1-FUDS
    # worker_evaluate(model=modds[2], devise=devices[0], X=X[p], Y=Y[p],
    #                 file=c_files[2], attempt=a, profile=profiles[p])

    # GPU DST model based
    # for i in range(0, len(gpu_threads)):
    #! GPU 0 - Strongest
    # for model_type in range(0, len(modds)):
    cycle_type = 0 # DST
    gpu_threads[0] = threading.Thread(target=worker_evaluate,
        args=[model, devices[1], X[cycle_type], Y[cycle_type],
            c, a, profiles[cycle_type]]
        )
    gpu_threads[0].start()
    # print('GPU0 - kicks off')
    #! GPU 1 - Weaker
    cycle_type = 1 # US06
    gpu_threads[1] = threading.Thread(target=worker_evaluate,
        args=[model, devices[2], X[cycle_type], Y[cycle_type],
            c, a, profiles[cycle_type]]
        )
    gpu_threads[1].start()
    # print('GPU1 - kicks off')
    #! CPU - Weakest/Middle
    cycle_type = 2 # FUDS
    # multi_cpu(model=modds[model_type], device='/cpu:0', n_threads=3,
    #             X=X[cycle_type], Y=Y[cycle_type],
    #             file=c_files[model_type], attempt=a, profile=profiles[cycle_type])
    cpu_thread = threading.Thread(target=multi_cpu,
            args=[model, '/cpu:0', 3,
                X[cycle_type], Y[cycle_type],
                c, a, profiles[cycle_type]]
        )
    cpu_thread.start()
    # print('CPU - kicks off')
    cpu_thread.join()
    for thread in gpu_threads:
        # print('GPU - join')
        thread.join()
    print(f'> > {c} at {a} completed')
    print(f'> > > Attempt {a} completed')
    # for m in range(0, len(modds)):
    #     multi_cpu(model=modds[m], device='/cpu:0', n_threads=3,
    #                 X=X[1], Y=Y[1],
    #                 file=c_files[2], attempt=a, profile=profiles[1])
    
    # Use the data split of 38.5/61.5
    # distributed_strategy(model=modds[0], devices=devices[1:], X=X[p], Y=Y[p],
    #                 file=c_files[0], attempt=a, profile=profiles[p])

    # with tf.device(devices[1]):
    #     metrics = valid_loop(model=modds[0], dist_input=(X[0], Y[0]), verbose=1)

    # client.execute(
    #     f'INSERT INTO {database}.{accuracies} (File, Attempt, Profile, mae, rms, r_s) VALUES ',
    #     [{'File' : c_files[0], 'Attempt' : a, 'Profile' : profiles[0],
    #       'mae' : metrics[0], 'rms' : metrics[1], 'r_s' : metrics[2]}]
    # )

    # del modds
    gc.collect()
    del model
    os.remove(f'{tmp_folder}model')
    #* Take a rest for a minute
    print('Taking a minute rest')
    time.sleep(60)
    # break
# Collect garbage leftovers
# %%
# from paramiko import SSHClient, AutoAddPolicy
# from scp import SCPClient
# from getpass import getpass
# def createSSHClient(server : str, port : int,
#                     user : str, password : str) -> SSHClient:
#     client : SSHClient = SSHClient()
#     client.load_system_host_keys()
#     client.set_missing_host_key_policy(AutoAddPolicy())
#     client.connect(hostname = server, port = port,
#                    username = user, password = password)
#     return client

# def progress4(filename, size, sent, peername) -> None:
#     sys.stdout.write(
#         "(%s:%s) %s's progress: %.2f%%   \r" % (peername[0], peername[1],
#                                     filename, float(sent)/float(size)*100)
#         )
# def get_meta(c_file) -> tuple:
#     return client.execute(
# f"""
# SELECT
#     ModelID,
#     Name,
#     Attempt,
#     Profile
# FROM ml_base.mapper
# WHERE File = {c_file}
# """
#     )[0]
# def get_epoch(c_file, metric) -> int:
#     return client.execute(
# f"""
# SELECT Epoch
# FROM ml_base.histories
# WHERE (File = {c_file}) AND ({metric} = (
#     SELECT MIN({metric})
#     FROM ml_base.histories
#     WHERE File = {c_file}
# ))
# """
#     )[0][0]
# username : str = 'n9312706' # getpass(prompt='QUT ID: ', stream=None)

# ssh = createSSHClient(
#         server='lyra.qut.edu.au',
#         port=22,
#         user=username,
#         password=getpass(prompt='Password: ', stream=None)
#     )

# remote_path : str = f'/mnt/home/{username}/MPhil/TF/Battery_SoCv4/Mods/'

# scp = SCPClient(ssh.get_transport(), progress4=progress4)

# meta = get_meta(c_file=c)
# assoc : dict = {'Chemali2017' : 1, 'BinXiao2020' : 2, 'TadeleMamo2020' : 3}
# folder_path = f'ModelsUp-{assoc[meta[1]]}/3x{meta[1]}-(131)/{meta[2]}-{meta[3]}/'

# epoch = get_epoch(c_file=c, metric='mae')
# epoch = 25
# path = remote_path + folder_path + str(epoch)

# scp.get(path, f'{tmp_folder}25-FUDS.h5')

# tmpmodel = tf.keras.models.load_model(f'{tmp_folder}60', compile=False)
# # model = create_model(
# #             tf.keras.layers.LSTM, layers=3, neurons=131,
# #             dropout=0.0, input_shape=X[2].shape[-2:], batch=1
# #         )
# model = tf.keras.models.load_model(f'{tmp_folder}19', compile=False)
# model.set_weights(weights=tmpmodel.get_weights())
# model.compile(
#     loss=tf.losses.MeanAbsoluteError(),
#     optimizer=tf.optimizers.SGD(0.001),
#     metrics=[tf.metrics.MeanAbsoluteError(),
#                 tf.metrics.RootMeanSquaredError(),
#                 tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
#     )
# with tf.device('/cpu:0'):
#     # metrics = valid_loop(model=model, dist_input=(X[2],Y[2]), verbose=1)
#     metrics = model.evaluate(x=X[2][:,0,:,:],y=Y[2][:,0],
#                 batch_size=1, verbose=1,
#                 sample_weight=None, steps=None, callbacks=None,
#                 max_queue_size=10, workers=1,
#                 use_multiprocessing=True, return_dict=True)
    # pred = model.predict(X[2][:10000,0,:,:], batch_size=1)