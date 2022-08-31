# %%
from pydoc import cli
from typing import Tuple
from clickhouse_driver import Client
import pandas as pd
import re, os, sys
import time

from py_modules.utils import Locate_Best_Epoch

from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient
from getpass import getpass

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

# %%
#? 1) Mapper table ['File', 'ModelID', 'Profile', 'Attempt', 'Name' 'Hist_plots(train, valid)',
#?                                                    'Best_plots(train, valid, test1, test2)']
#? 2) Histories table ['File', 'Epoch', 'mae', 'rmse', 'rsquare', 'time_s', 'learn_r',
#?                                                                       'train()', 'val()', 'tes()']
#? 3) Faulty Histories ['File', 'Epoch', 'attempt', 'mae', 'time_s', learn_rate, train ()]
#? 4) Logits table ['File', 'Epoch', logits()]
#? 5) Models table ['File', 'Epochs', 'Latest_Model', 'TR_Model', 'VL_Model', 'TS_Model']
#???## Sizes are same, time not a priority at training, trendines not ready 
columns : str = ''
mapper : str = 'mapper'
histores : str = 'histories'
faulties : str = 'faulties'
models : str = 'models'

# %%
#? 1) Mapper table
columns = (f'CREATE TABLE {database}.{mapper} ('
#                            "id UInt64, "
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
columns  = (f'CREATE TABLE {database}.{histores} ('
#                            "id UInt64, "
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
columns = (f'CREATE TABLE {database}.{faulties} ('
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
columns = (f'CREATE TABLE {database}.logits_train ('
#                            "id UInt64, "
                            "File UInt64, "
                            'Epoch Int32, '
                            'Logits Float32 '
                        ') '
                        'ENGINE = MergeTree() '
                        'ORDER BY (File, Epoch);'
                )
client.execute(columns)
columns = (f'CREATE TABLE {database}.logits_valid ('
#                            "id UInt64, "
                            "File UInt64, "
                            'Epoch Int32, '
                            'Logits Float32 '
                        ') '
                        'ENGINE = MergeTree() '
                        'ORDER BY (File, Epoch);'
                )
client.execute(columns)
columns = (f'CREATE TABLE {database}.logits_test ('
#                            "id UInt64, "
                            "File UInt64, "
                            'Epoch Int32, '
                            'Logits Float32 '
                        ') '
                        'ENGINE = MergeTree() '
                        'ORDER BY (File, Epoch);'
                )
client.execute(columns)

#? 5) Models table
columns = (f'CREATE TABLE {database}.{models} ('
#                            "id UInt64, "
                            "File UInt64, "
                            'Epochs Tuple('
                                'latest Int32, '
                                'tr_best Int32, '
                                'vl_best Int32, '
                                'ts_best Int32 '
                            '), '
                            'latest String, '
                            'train String, '
                            'valid String, '
                            'testi String '
                        ') '
                        'ENGINE = MergeTree() '
                        'ORDER BY (File);'
                )
client.execute(columns)

print(show_tables())
# %%
def query_mapper(file_N : int, model_loc : str, ModelID : int,
                 profile : str, attempt : int, name : str, best_epoch : int) -> str:
    #? 1) Mapper table ['File', 'ModelID', 'Profile', 'Attempt', 'Name' 'Hist_plots(train, valid)',
    #?                                                    'Best_plots(train, valid, test1, test2)']
    profs : list[str] = ['DST', 'US06', 'FUDS']
    order = ['File', 'ModelID', 'Profile', 'Attempt', 'Name', 'Hist_plots', 'Best_plots']
    order_str = ', '.join(order)

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

    query = (f'INSERT INTO {database}.{mapper} ({order_str}) VALUES ')
    query +=(f"("
             f"{file_N}, {ModelID}, '{profile}', {attempt}, '{name}', "
             f"('{train}', '{valid}'), ('{b_train}', '{b_valid}', '{b_test[0]}', '{b_test[1]}')"
            # f"('train', 'valid'), ('b_train', 'b_valid', 'b_test[0]','b_test[1]')"
             ")"
             ";"
            )
    return query

def prep_hist_frame(data : pd.DataFrame, file_n : int, dropNans : bool = True) -> pd.DataFrame:
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

def prep_faulty_frame(data : pd.DataFrame, file_n : int, dropNans : bool = True) -> pd.DataFrame:
    data = data.rename(columns={'time(s)' : 'time_s', 'learning_rate' : 'learn_r'})
    data.drop(['loss', 'train_l'], axis=1)
    data['File'] = file_n
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

def query_logits(file_N : int, model_loc : str, epoch : int,
                 table : str, stage : str) -> str:
                   # {i}-train-logits.csv
                # {i}-valid-logits.csv
                # {i}-test--logits.csv
    order = ['File', 'Epoch', 'Logits']
    order_str = ', '.join(order)
    data = pd.read_csv(f'{model_loc}{epoch}-{stage}-logits.csv',
                         names=['index', 'Logits'], header=None)[1:]
    data = data.reset_index()
    data = data.drop(['index'], axis=1)
    data['Epoch'] = epoch
    data['File'] = file_N
    query = (f'INSERT INTO {database}.logits_{table} ({order_str}) VALUES')
    for i in range(len(data)):
        cut = ', '.join((str(v) for v in data.loc[i,order].values))
        query += f'({cut}), '
    return query[:-2] + ';' # Cul the last space and comma and close query with ;

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

# %%
#names : list[str] = ['Chemali2017', 'BinXiao2020', 'TadeleMamo2020']
names : list[str] = ['TadeleMamo2020']
model_names : list[str] = [f'ModelsUp-{i}' for i in range(3,4)]
attempts : range = range(1,11)
# profiles : list[str] = ['DST', 'US06', 'FUDS']
profiles : list[str] = ['US06', 'FUDS']

file_name : str = '' # Name associated with ID
model_loc : str = '' # Location of files to extract data from
c_files : int = 71
re_faulty = re.compile(".*-faulty-history.csv")
for n, model_name in enumerate(model_names):
    file_name = names[n]
    for profile in profiles:
        for attempt in attempts:
            # if (file_name == 'BinXiao2020') and (attempt == 1) and (profile == 'DST'):
            #     continue
            model_loc : str = f'/home/sadykov/Remote_Battery_SoCv4/Battery_SoCv4/Mods/{model_name}/3x{file_name}-(131)/{attempt}-{profile}/'
            print(model_loc)
            # model_name = 'ModelsUp-2'
            # file_name = 'BinXiao2020'
            # attempt = 1
            # profile = 'FUDS'
            # model_loc = '/home/sadykov/Remote_Battery_SoCv4/Battery_SoCv4/Mods/ModelsUp-2/3xBinXiao2020-(131)/1-FUDS/'

            #* 1) Table 1 - Mapper
            b_Epoch, _ = Locate_Best_Epoch(f'{model_loc}history.csv', 'mae')
            query = query_mapper(file_N = c_files, model_loc=model_loc,
                                 ModelID=n+1, profile=profile, attempt=attempt,
                                 name = file_name, best_epoch=b_Epoch)
            client.execute(query=query)

            
            #* 2) Table 2 - Histories quering
            df_histories = pd.read_csv(f'{model_loc}history.csv')
            df_histories = prep_hist_frame(df_histories, c_files, dropNans=True)
            query = query_hist_frame(df_histories)
            client.execute(query=query)

            #* 3) Table 3 - Faulties
            ## Identify faulty histories.            
            # 5-faulty-history.csv
            
            for file in list(filter(re_faulty.match,
                                    os.listdir(f'{model_loc}'))):
                df_faulties = pd.read_csv(f'{model_loc}{file}')
                df_faulties = prep_faulty_frame(df_faulties, c_files, dropNans=False)
                query = query_faulty_frame(df_faulties)
                client.execute(query=query)

            #* 4) Table 4 Logits
            ## loop through every epoch
            #! Optimise this. possible make 12K on horizontal, rather than vertical
            for epoch in range(1, b_Epoch+1):
                query = query_logits(file_N=c_files, model_loc=model_loc,
                                     epoch=epoch, table='train', stage='train')
                client.execute(query=query)
                query = query_logits(file_N=c_files, model_loc=model_loc,
                                     epoch=epoch, table='valid', stage='valid')
                client.execute(query=query)
                query = query_logits(file_N=c_files, model_loc=model_loc,
                                     epoch=epoch, table='test', stage='test-')
                client.execute(query=query)

            c_files +=1
    #         break
    #     break
    # break

# %%
username : str = 'n9312706' # getpass(prompt='QUT ID: ', stream=None)

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

# scp.put('test.txt', 'test2.txt')
# scp.get('test2.txt')
# # Uploading the 'test' directory with its content in the
# # '/home/user/dump' remote directory
# scp.put('test', recursive=True, remote_path='/home/user/dump')

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

def get_epoch(c_file, metric) -> int:
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
# %%
# describe_table(histores)
# df = client.query_dataframe("SELECT * FROM ml_base.histories where File == 1;")
tmp_folder = 'Modds/tmp/'
m_names = ['best', 'trai', 'vali', 'test']
#* 1) Locate how many files stored
c_files = client.execute('SELECT MAX(File) FROM ml_base.mapper;')[0][0]
#* 2) For every file do
c = 1
assoc : dict = {'Chemali2017' : 1, 'BinXiao2020' : 2, 'TadeleMamo2020' : 3}
for c in range(1,c_files+1):
    #* 2.1) Get the details and sort the path
    meta = get_meta(c_file=c)
    folder_path = f'ModelsUp-{assoc[meta[1]]}/3x{meta[1]}-(131)/{meta[2]}-{meta[3]}/'
    print(meta)
    #* 2.2) Get the history in DF format to work with
    epochs = []
    for metric in ['mae', 'train.mae', 'val.mae', 'tes.mae']:
        epochs.append(get_epoch(c_file=c, metric=metric))

    
    #* 2.3) Getting the bests and save to tmp
    for i, best_epoch in enumerate(epochs):
        path = remote_path + folder_path + str(best_epoch)
        scp.get(path, f'{tmp_folder}{m_names[i]}')
    
    # l_models = []
    # for name in m_names:
    #     with open(f'{tmp_folder}{name}', 'rb') as file:
    #         l_models.append(file.read())
    with open(f'{tmp_folder}{m_names[0]}', 'rb') as file:
        ml_best = file.read()
    with open(f'{tmp_folder}{m_names[1]}', 'rb') as file:
        ml_trai = file.read()
    with open(f'{tmp_folder}{m_names[2]}', 'rb') as file:
        ml_vali = file.read()
    with open(f'{tmp_folder}{m_names[3]}', 'rb') as file:
        ml_test = file.read()

    client.execute(
        f'INSERT INTO {database}.{models} (File, Epochs, latest, train, valid, testi) VALUES ',
        [{'File' : c, 'Epochs' : tuple(epochs), 'latest' : ml_best,
          'train' : ml_trai, 'valid' : ml_vali, 'testi' : ml_test}]
    )

    for name in m_names:
        os.remove(f'{tmp_folder}{name}')

    del epochs
    del ml_best, ml_trai, ml_vali, ml_test

#* Store in a table....

# %%
#? 5) Models table

# describe_table(models)
# drop_table(models)
# %%
scp.close()
ssh.close()