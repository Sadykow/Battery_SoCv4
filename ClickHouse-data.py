# %%
import uuid
from clickhouse_driver import Client
import pandas as pd
import re, os, sys, threading
import time

import tensorflow as tf  # Tensorflow and Numpy replacement
import tensorflow_addons as tfa
from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
from py_modules.Attention import *

from py_modules.utils import Locate_Best_Epoch

from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient
from getpass import getpass

from itertools import chain
from numpy import ndarray
from tqdm import trange
from typing import Callable

import gc
# %%
physical_devices = tf.config.experimental.list_physical_devices()
print(f'Available devices:\n{physical_devices}')

for gpu in physical_devices[1:]:
    tf.config.experimental.set_memory_growth(
                                gpu, False)
# %%
database = 'ml_base'
client = Client(host='192.168.1.254', port=9000, 
                database=database, user='tf', password='TF28')

client.execute('SHOW DATABASES')
# %%
def show_tables():
    return client.execute(f'SHOW TABLES FROM {database};')
def drop_table(table : str):
    return client.execute(f'DROP TABLE IF EXISTS {database}.{table};')
def describe_table(table : str):
    return client.execute(f'DESCRIBE TABLE {database}.{table} SETTINGS describe_include_subcolumns=1;')
def get_table(table : str):
    return client.execute(f'SELECT * FROM {database}.{table}')
def create_uuid():
    return client.execute('SELECT generateUUIDv4();')[0][0]

# %%
#? 1) Mapper table ['File', 'ModelID', 'Profile', 'Attempt', 'Name' 'Hist_plots(train, valid)',
#?                                                    'Best_plots(train, valid, test1, test2)']
#? 2) Histories table ['File', 'Epoch', 'mae', 'rmse', 'rsquare', 'time_s', 'learn_r',
#?                                                                       'train()', 'val()', 'tes()']
#? 3) Faulty Histories ['File', 'Epoch', 'attempt', 'mae', 'time_s', learn_rate, train ()]
#? 4) Logits table logits_(train, valid,test) ['File', 'Epoch', logits()]
#? 5) Models table ['File', 'Epochs', 'Latest_Model', 'TR_Model', 'VL_Model', 'TS_Model']
#? 6) Full train logits table ['File', 'Epochs', 'Logits']
#???## Sizes are same, time not a priority at training, trendines not ready 
columns : str = ''
mapper : str = 'mapper'
histores : str = 'histories'
faulties : str = 'faulties'
models : str = 'models'
accuracies : str = 'logits_full_train'
# %%
#? 1) Mapper table
columns = (f'CREATE TABLE IF NOT EXISTS {database}.{mapper} ('
                            "id UUID, "
                            "File UInt64, "
                            'ModelID UInt32, '
                            'Profile String, '
                            'Attempt UInt32, '
                            'Name String, '
                            'Hist_plots Tuple('
                                'train String, '
                                'valid String'
                            '), '
                            'Best_plots Tuple('
                                'train String, '
                                'valid String, '
                                'test1 String, '
                                'test2 String '
                            ') '
                        ') '
                        'ENGINE = MergeTree() '
                        'ORDER BY (File, ModelID, Attempt);'
                )
client.execute(columns)

#? 2) Histories table
columns  = (f'CREATE TABLE IF NOT EXISTS {database}.{histores} ('
                            "id UUID, "
                            "File UInt64, "
                            'Epoch Int32, '
                            'mae Float32, '
                            'rmse Float32, '
                            'rsquare Float32, '
                            'time_s Float32, '
                            'learn_r Float32, '
                            'train Tuple('
                                'mae Float32, '
                                'rms Float32, '
                                'r_s Float32 '
                            '), '
                            'val Tuple('
                                'mae Float32, '
                                'rms Float32, '
                                'r_s Float32, '
                                't_s Float32 '
                            '), '
                            'tes Tuple('
                                'mae Float32, '
                                'rms Float32, '
                                'r_s Float32, '
                                't_s Float32 '
                            ') '
                        ') '
                        'ENGINE = MergeTree() '
                        'ORDER BY (File, Epoch);'
                )
client.execute(columns)

#? 3) Faulties table
columns = (f'CREATE TABLE IF NOT EXISTS {database}.{faulties} ('
#                            "id UInt64, "
                            "File UInt64, "
                            'Epoch Int32, '
                            'attempt Float32, '
                            'mae Float32, '
                            'time_s Float32, '
                            'learn_r Float32, '
                            'train Tuple('
                                'mae Float32, '
                                'rms Float32, '
                                'r_s Float32 '
                            ') '
                        ') '
                        'ENGINE = MergeTree() '
                        'ORDER BY (File, Epoch);'
                )
client.execute(columns)

#? 4) Logits table
columns = (f'CREATE TABLE IF NOT EXISTS {database}.logits_train ('
                            "id UUID, "
                            "File UInt64, "
                            'Epoch Int32, '
                            'Logits Array(Float32) '
                        ') '
                        'ENGINE = MergeTree() '
                        'ORDER BY (File, Epoch);'
                )
client.execute(columns)
columns = (f'CREATE TABLE IF NOT EXISTS {database}.logits_valid ('
                            "id UUID, "
                            "File UInt64, "
                            'Epoch Int32, '
                            'Logits Array(Float32) '
                        ') '
                        'ENGINE = MergeTree() '
                        'ORDER BY (File, Epoch);'
                )
client.execute(columns)
columns = (f'CREATE TABLE IF NOT EXISTS {database}.logits_test ('
                            "id UUID, "
                            "File UInt64, "
                            'Epoch Int32, '
                            'Logits Array(Float32) '
                        ') '
                        'ENGINE = MergeTree() '
                        'ORDER BY (File, Epoch);'
                )
client.execute(columns)

#? 5) Models table
columns = (f'CREATE TABLE IF NOT EXISTS {database}.{models} ('
                            "id UUID, "
                            "File UInt64, "
                            'Epochs Tuple('
                                'latest Int32, '
                                'tr_best Int32, '
                                'vl_best Int32, '
                                'ts_best Int32 '
                            '), '
                            'latest String, '
                            'train String, ' #! Get the working one
                            'valid String, '
                            'testi String '
                        ') '
                        'ENGINE = MergeTree() '
                        'ORDER BY (File);'
                )
client.execute(columns)

#? 6) Full train Logits
columns = (f'CREATE TABLE IF NOT EXISTS {database}.{accuracies} ('
                            "id UUID, "
                            "File UInt64, "
                            'Attempt UInt32, '
                            'Profile String, '
                            # 'mae Float32, '
                            # 'rms Float32, '
                            # 'r_s Float32 '
                            'Logits Array(Float32)'
                        ') '
                        'ENGINE = MergeTree() '
                        'ORDER BY (File, Attempt, Profile);'
                )
client.execute(columns)

print(show_tables())
# %%
def insert_mapper(UUID : uuid, file_N : int, model_loc : str, ModelID : int,
                 profile : str, attempt : int, name : str, best_epoch : int) -> str:
    #? 1) Mapper table ['File', 'ModelID', 'Profile', 'Attempt', 'Name' 'Hist_plots(train, valid)',
    #?                                                    'Best_plots(train, valid, test1, test2)']
    profs : list[str] = ['DST', 'US06', 'FUDS']
    order = ['id', 'File', 'ModelID', 'Profile', 'Attempt', 'Name',
                 'Hist_plots', 'Best_plots']
    # order_str = ', '.join(order)

    # Histories plots train and valid
    with open(f'{model_loc}history-{profile}-train.svg', 'rb') as file:
        train = file.read().decode('utf-8')
    with open(f'{model_loc}history-{profile}-valid.svg', 'rb') as file:
        valid = file.read().decode('utf-8')

    # Best plots for train, valid, test1 and test2
    with open(f'{model_loc}traiPlots/{profile}-tra-{best_epoch}.svg', 'rb') as file:
        b_train = file.read().decode('utf-8')
    with open(f'{model_loc}valdPlots/{profile}-val-{best_epoch}.svg', 'rb') as file:
        b_valid = file.read().decode('utf-8')
    
    profs.remove(profile)
    b_test : list = []
    for p in profs:
        with open(f'{model_loc}testPlots/{profile}-{p}-{best_epoch}.svg', 'rb') as file:
            b_test.append(file.read().decode('utf-8'))

    query = (f'INSERT INTO {database}.{mapper} VALUES ')
    params = {
            'id' : UUID, 'File' : file_N, 'ModelID' : ModelID,
            'Profile' : profile, 'Attempt' : attempt, 'Name' : name,
            'Hist_plots' : {
                    'train' : train,
                    'valid' : valid
                },
            'Best_plots' : {
                    'train' : b_train,
                    'valid' : b_valid,
                    'test1' : b_test[0],
                    'test2' : b_test[1]
                }
        }
    try:
        return client.execute(
            query=query,
            params=[params]
            )
    except Exception:
        print(f'\n>>>Failed Execution {file_N}\n\n')
        return (query, [params[key] for key in order[:6]])

    # query = (f'INSERT INTO {database}.{mapper} ({order_str}) VALUES ')
    # query +=(f"("
    #          f"'{UUID}', {file_N}, {ModelID}, '{profile}', {attempt}, '{name}', "
    #          f"('{train}', '{valid}'), ('{b_train}', '{b_valid}', '{b_test[0]}', '{b_test[1]}')"
    #         # f"('train', 'valid'), ('b_train', 'b_valid', 'b_test[0]','b_test[1]')"
    #          ")"
    #          ";"
    #         )
    # return query

def prep_hist_frame(data : pd.DataFrame, uuID : uuid, file_n : int, dropNans : bool = True) -> pd.DataFrame:
    """ Preprocesses the history data by merging columns into a tuple and removes unused

    Args:
        data (pd.DataFrame): _description_
        file_n (int): _description_
        dropNans (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    # Alter columns
    data = data.drop(['loss', 'train_l', 'vall_l', 'test_l'], axis=1)
    data = data.rename(columns={'time(s)' : 'time_s'})
    data['File'] = file_n
    data['id'] = uuID

    # Compress the columnss
    data['train'] = list(zip(data['train_mae'], data['train_rms'], data['train_r_s']))
    data = data.drop(['train_mae', 'train_rms', 'train_r_s'], axis=1)
    data['val'] = list(zip(data['val_mae'], data['val_rms'], data['val_r_s'], data['val_t_s']))
    data = data.drop(['val_mae', 'val_rms', 'val_r_s', 'val_t_s'], axis=1)
    data['tes'] = list(zip(data['tes_mae'], data['tes_rms'], data['tes_r_s'], data['tes_t_s']))
    data = data.drop(['tes_mae', 'tes_rms', 'tes_r_s', 'tes_t_s'], axis=1)

    if (dropNans):
        return data.dropna()
    else:
        return data

def prep_faulty_frame(data : pd.DataFrame, uuID : uuid, file_n : int, dropNans : bool = True) -> pd.DataFrame:
    data = data.rename(columns={'time(s)' : 'time_s', 'learning_rate' : 'learn_r'})
    data.drop(['loss', 'train_l'], axis=1)
    data['File'] = file_n
    data['id'] = uuID

    data['train'] = list(zip(data['train_mae'], data['train_rms'], data['train_r_s']))
    data = data.drop(['train_mae', 'train_rms', 'train_r_s'], axis=1)
    if(len(data[data['Epoch'] =='Epoch'])>0):
        data = data[data['Epoch'] !='Epoch']
        data = data.reset_index()
        data = data.drop(['index'], axis=1)
    if (dropNans):
        return data.dropna()
    else:
        return data

def query_hist_frame(data : pd.DataFrame) -> str:
    order = ['File', 'Epoch', 'mae', 'rmse', 'rsquare', 'time_s', 'learn_r', 'train', 'val', 'tes']
    order_str = ', '.join(order)
    query = (f'INSERT INTO {database}.{histores} ({order_str}) VALUES')
    for i in range(len(data)):
        cut = ', '.join((str(v) for v in data.loc[i,order].values))
        query += f'({cut}), '
    return query[:-2] + ';' # Cul the last space and comma and close query with ;

def query_faulty_frame(data : pd.DataFrame) -> str:
    order = ['File', 'Epoch', 'attempt', 'mae', 'time_s', 'learn_r', 'train']
    order_str = ', '.join(order)
    query = (f'INSERT INTO {database}.{faulties} ({order_str}) VALUES')
    for i in range(len(data)):
        cut = ', '.join((str(v) for v in data.loc[i,order].values))
        query += f'({cut}), '
    return query[:-2] + ';' # Cul the last space and comma and close query with ;

def iter_hist_frame(df):
    # order = ['id', 'File', 'Epoch', 'mae', 'rmse', 'rsquare', 'time_s', 'learn_r',
            #  'train', 'val', 'tes']
    for i in range(len(df)):
        yield df.loc[i,:].to_dict()

def insert_logits(uuID : uuid, file_N : int, model_loc : str, epoch : int,
                 table : str, stage : str) -> str:
                   # {i}-train-logits.csv
                # {i}-valid-logits.csv
                # {i}-test--logits.csv
    order = ['id', 'File', 'Epoch', 'Logits']
    order_str = ', '.join(order)
    data = pd.read_csv(f'{model_loc}{epoch}-{stage}-logits.csv',
                         names=['index', 'Logits'], header=None)[1:]
    data = data.reset_index()
    data = data.drop(['index'], axis=1)
    # client.execute(
    #         f'INSERT INTO {database}.logits_{table} ({order_str}) VALUES',
    #         [{'File' : file_N, 'Epoch' : epoch,
    #         'Logits' : data['Logits']}]
    #     )
    
    query = f'INSERT INTO {database}.logits_{table} ({order_str}) VALUES'
    params = {
            'id' : uuID, 'File' : file_N,
            'Epoch' : epoch, 'Logits' : data['Logits']
        }
    try:
        return client.execute(
            query=query,
            params=[params]
            )
    except Exception:
        print(f'\n>>>Failed Execution {file_N}\n\n')
        return (query, [params[key] for key in order[:3]])

    # query = (f'INSERT INTO {database}.logits_{table} ({order_str}) VALUES')
    # for i in range(len(data)):
    #     cut = ', '.join((str(v) for v in data.loc[i,order].values))
    #     query += f'({cut}), '
    # return query[:-2] + ';' # Cul the last space and comma and close query with ;

def get_min_epoch(c_file, metric) -> int:
    return client.execute(
f"""
SELECT Epoch
FROM ml_base.histories
WHERE (File = {c_file}) AND ({metric} = (
    SELECT MIN({metric})
    FROM ml_base.histories
    WHERE File = {c_file}
))
"""
    )[0][0]

def get_healthy_epoch(c_file) -> int:
    return client.execute(
f"""
SELECT MIN(Epoch)
FROM ml_base.faulties
WHERE File = {c_file}
"""
    )[0][0]
# %%
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

proto : str = 'smb'
if(proto == 'ssh'):
    ssh = createSSHClient(
            server='lyra.qut.edu.au',
            port=22,
            user=username,
            password=getpass(prompt='Password: ', stream=None)
        )
        
    # stdin, stdout, stderr = ssh.exec_command('ls')
    # out = stdout.read().decode().strip()
    # error = stderr.read().decode().strip()

    remote_path : str = f'/mnt/home/{username}/MPhil/TF/Battery_SoCv4/Mods/'

    scp = SCPClient(ssh.get_transport(), progress4=progress4)
elif (proto == 'smb'):
    remote_path : str = f'/mnt/Lyra/MPhil/TF/Battery_SoCv4/Mods/'
else:
    print('Which hell of the protocol is in use')
# scp.put('test.txt', 'test2.txt')
# scp.get('test2.txt')
# # Uploading the 'test' directory with its content in the
# # '/home/user/dump' remote directory
# scp.put('test', recursive=True, remote_path='/home/user/dump')
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
# %%
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
                    X:ndarray, Y:ndarray, uuID : uuid,
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
            '(id, File, Attempt, Profile, Logits) VALUES ',
            [{'id' : uuID, 'File' : file, 'Attempt' : attempt, 'Profile' : profile,
            'Logits' : metrics}]
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
def multi_cpu(model, device : str, n_threads : int,
                    X:ndarray, Y:ndarray, uuID : uuid,
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
            '(id, File, Attempt, Profile, Logits) VALUES ',
            [{'id' : uuID, 'File' : file, 'Attempt' : attempt, 'Profile' : profile,
            'Logits' : list(chain.from_iterable(metrics))}]
        )
    print(f'>{device} Worker: {file} on {attempt}x{profile} is over in {toc}!',
          flush=True)
# %%
#names : list[str] = ['Chemali2017', 'BinXiao2020', 'TadeleMamo2020']
names : list[str] = ['MengJiao2020']
assoc : dict = {'Chemali2017' : 1, 'BinXiao2020' : 2, 'TadeleMamo2020' : 3,
                'GelarehJavid2020' : 4, 'MengJiao2020' : 5}
# model_names : list[str] = [f'ModelsUp-{i}' for i in range(3,4)]
attempts : range = range(1,11)
profiles : list[str] = ['DST', 'US06', 'FUDS']
# profiles : list[str] = ['US06', 'FUDS']

devices : list = ['/cpu:0','/gpu:0','/gpu:1']
cpu_thread : threading.Thread = None
gpu_threads  : list = [threading.Thread]*2

file_name : str = names[0] # Name associated with ID
model_loc : str = '' # Location of files to extract data from
c_files : int = client.execute(f'SELECT COUNT(*) FROM {database}.{mapper};')[0][0]
re_faulty = re.compile(".*-faulty-history.csv")
profile = 'DST'
attempt = 1
for file_name in names:
    # file_name = names[n]
    model_name = f'ModelsUp-{assoc[file_name]}'
    for profile in profiles:
        for attempt in attempts:
            if (file_name == 'TadeleMamo2020') and (attempt == 1) and (profile == 'DST'):
                continue
            # model_loc : str = (f'/home/sadykov/Remote_Battery_SoCv4/Battery_SoCv4/'
            #             f'Mods/{model_name}/3x{file_name}-(131)/{attempt}-{profile}/')
            model_loc : str = (f'{remote_path}'
                        f'{model_name}/3x{file_name}-(131)/{attempt}-{profile}/')
            print(model_loc)
            # model_name = 'ModelsUp-2'
            # file_name = 'BinXiao2020'
            # attempt = 1
            # profile = 'FUDS'
            # model_loc = '/home/sadykov/Remote_Battery_SoCv4/Battery_SoCv4/Mods/ModelsUp-2/3xBinXiao2020-(131)/1-FUDS/'
            #* 0) Create a unique associated ID
            uuID = create_uuid()
            c_files +=1

            #* 1) Table 1 - Mapper
            b_Epoch, _ = Locate_Best_Epoch(f'{model_loc}history.csv', 'mae')
            insert_mapper(UUID=uuID, file_N = c_files, model_loc=model_loc,
                                ModelID=assoc[file_name], profile=profile, attempt=attempt,
                                name = file_name, best_epoch=b_Epoch)
        
            
            #* 2) Table 2 - Histories quering
            df_histories = pd.read_csv(f'{model_loc}history.csv')
            # df_histories = prep_hist_frame(df_histories, uuID, c_files, dropNans=True)
            # client.execute(query=query_hist_frame(df_histories))
            client.execute(query=f'INSERT INTO {database}.{histores} VALUES',
                           params=iter_hist_frame(
                                    df=prep_hist_frame(df_histories, uuID,
                                                    c_files, dropNans=True)
                                )
                        )

            #* 3) Table 3 - Faulties
            ## Identify faulty histories.            
            # 5-faulty-history.csv
            
            for file in list(filter(re_faulty.match,
                                    os.listdir(f'{model_loc}'))):
                df_faulties = pd.read_csv(f'{model_loc}{file}')
                # df_faulties = prep_faulty_frame(df_faulties, uuID,
                #                                  c_files, dropNans=False)
                # query = query_faulty_frame(df_faulties)
                client.execute(
                            query=f'INSERT INTO {database}.{faulties} VALUES',
                            params=iter_hist_frame(
                                    df=prep_faulty_frame(df_faulties, uuID,
                                                 c_files, dropNans=False)
                                )
                        )

            #* 4) Table 4 Logits
            ## loop through every epoch
            #! Optimise this. possible make 12K on horizontal, rather than vertical
            for epoch in range(1, b_Epoch+1):
                insert_logits(uuID=uuID, file_N=c_files, model_loc=model_loc,
                                     epoch=epoch, table='train', stage='train')
                insert_logits(uuID=uuID, file_N=c_files, model_loc=model_loc,
                                     epoch=epoch, table='valid', stage='valid')
                insert_logits(uuID=uuID, file_N=c_files, model_loc=model_loc,
                                     epoch=epoch, table='test', stage='test-')
                
            #* 5) Table 5 Models
            #** 5.1) Get epochs to drag models from
            epochs = []
            epochs.append(get_healthy_epoch(c_file=c_files)-1)
            for metric in ['train.mae', 'val.mae', 'tes.mae']:
                epochs.append(get_min_epoch(c_file=c_files, metric=metric))
            
            #** 5.2) Getting the bests and save
            mls = []
            for i, best_epoch in enumerate(epochs):
                path = model_loc + str(best_epoch)
                with open(path, 'rb') as file:
                    mls.append(file.read())
            client.execute(
                f'INSERT INTO {database}.{models} (id, File, Epochs, latest, train, valid, testi) VALUES ',
                [{'id' : uuID, 'File' : c_files, 'Epochs' : tuple(epochs),
                'latest' : mls[0], 'train' : mls[1], 'valid' : mls[2], 'testi' : mls[3]}]
            )

            #* 6) Compute the logits
            if(file_name == 'TadeleMamo2020'):
                model : tf.keras.models.Sequential = tf.keras.models.load_model(
                    model_loc + str(epochs[0]),
                    custom_objects={'AttentionWithContext' : AttentionWithContext,
                                    'Addition' : Addition},
                    compile=False)
            else:
                model = tf.keras.models.load_model(model_loc + str(epochs[0]), compile=False)
            cycle_type = 0 # DST
            gpu_threads[0] = threading.Thread(target=worker_evaluate,
                args=[model, devices[1], X[cycle_type], Y[cycle_type],
                    uuID, c_files, attempt, profiles[cycle_type]]
                )
            gpu_threads[0].start()
            # print('GPU0 - kicks off')
            #! GPU 1 - Weaker
            cycle_type = 1 # US06
            gpu_threads[1] = threading.Thread(target=worker_evaluate,
                args=[model, devices[2], X[cycle_type], Y[cycle_type],
                    uuID, c_files, attempt, profiles[cycle_type]]
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
                        uuID, c_files, attempt, profiles[cycle_type]]
                )
            cpu_thread.start()
            # print('CPU - kicks off')
            cpu_thread.join()
            for thread in gpu_threads:
                # print('GPU - join')
                thread.join()
            print(f'> > {c_files} at {attempt} completed')
            
            #* Clean up the stage 5-6
            del epochs, mls, model
            gc.collect()
            print('Taking a minute rest')
            time.sleep(60)
    #         break
    #     break
    # break
print("Meng finished - restore attempts and names")
# %%
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
# # %%
# # describe_table(histores)
# # df = client.query_dataframe("SELECT * FROM ml_base.histories where File == 1;")
# tmp_folder = 'Modds/tmp/'
# m_names = ['best', 'trai', 'vali', 'test']
# #* 1) Locate how many files stored
# c_files = client.execute('SELECT MAX(File) FROM ml_base.mapper;')[0][0]
# #* 2) For every file do
# c = 1
# assoc : dict = {'Chemali2017' : 1, 'BinXiao2020' : 2, 'TadeleMamo2020' : 3}
# for c in range(1,c_files+1):
#     #* 2.1) Get the details and sort the path
#     meta = get_meta(c_file=c)
#     folder_path = f'ModelsUp-{assoc[meta[1]]}/3x{meta[1]}-(131)/{meta[2]}-{meta[3]}/'
#     print(meta)
#     #* 2.2) Get the history in DF format to work with
#     epochs = []
#     for metric in ['mae', 'train.mae', 'val.mae', 'tes.mae']:
#         epochs.append(get_epoch(c_file=c, metric=metric))

    
#     #* 2.3) Getting the bests and save to tmp
#     for i, best_epoch in enumerate(epochs):
#         path = remote_path + folder_path + str(best_epoch)
#         scp.get(path, f'{tmp_folder}{m_names[i]}')
    
#     # l_models = []
#     # for name in m_names:
#     #     with open(f'{tmp_folder}{name}', 'rb') as file:
#     #         l_models.append(file.read())
#     with open(f'{tmp_folder}{m_names[0]}', 'rb') as file:
#         ml_best = file.read()
#     with open(f'{tmp_folder}{m_names[1]}', 'rb') as file:
#         ml_trai = file.read()
#     with open(f'{tmp_folder}{m_names[2]}', 'rb') as file:
#         ml_vali = file.read()
#     with open(f'{tmp_folder}{m_names[3]}', 'rb') as file:
#         ml_test = file.read()

#     client.execute(
#         f'INSERT INTO {database}.{models} (File, Epochs, latest, train, valid, testi) VALUES ',
#         [{'File' : c, 'Epochs' : tuple(epochs), 'latest' : ml_best,
#           'train' : ml_trai, 'valid' : ml_vali, 'testi' : ml_test}]
#     )

#     for name in m_names:
#         os.remove(f'{tmp_folder}{name}')

#     del epochs
#     del ml_best, ml_trai, ml_vali, ml_test

#* Store in a table....

# %%
#? 5) Models table

# describe_table(models)
# drop_table(models)
# %%
# scp.close()
# ssh.close()