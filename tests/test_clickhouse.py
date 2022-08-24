# %%
from clickhouse_driver import Client
import pandas as pd
import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt

import time
# %%
# with Client(host='192.168.1.254', port=9000, 
#                 database='ml_base', user='tf', password='TF28') as client:
#     client.execute('SHOW DATABASES')
database = 'ml_base'
client = Client(host='192.168.1.254', port=9000, 
                database=database, user='tf', password='TF28')

client.execute('SHOW DATABASES')

# %%
def show_tables():
    return f'SHOW TABLES FROM {database};'
def drop_table(table : str):
    return f'DROP TABLE IF EXISTS {database}.{table};'
def describe_table(table : str):
    return f'DESCRIBE TABLE {database}.{table} SETTINGS describe_include_subcolumns=1;'
def get_table(table : str):
    return f'SELECT * FROM {database}.{table}'

# %%
# histores_name = 'histories'
# histories_columns : str = (f'CREATE TABLE {database}.{histores_name} ('
# #                            "id UInt64, "
#                             "File UInt64, "
#                             'Epoch Int32, '
#                             'mae Float32, '
#                             'rmse Float32, '
#                             'rsquare Float32, '
#                             'time_s Float32, '
#                             'train_mae Float32, '
#                             'train_rms Float32, '
#                             'train_r_s Float32, '
#                             'val_mae Float32, '
#                             'val_rms Float32, '
#                             'val_r_s Float32, '
#                             'val_t_s Float32, '
#                             'tes_mae Float32, '
#                             'tes_rms Float32, '
#                             'tes_r_s Float32, '
#                             'tes_t_s Float32, '
#                             'learn_r Float32'
#                         ') '
#                         'ENGINE = MergeTree() '
#                         'ORDER BY (File);'
#                 )
# print(histories_columns)
# client.execute(histories_columns)

# %%
#? 1) Table 1 with all histories to store
histores_name = 'histories'
histories_columns : str = (f'CREATE TABLE {database}.{histores_name} ('
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
# print(histories_columns)
client.execute(histories_columns)

#? 2) Table 2 with asociations and categories and BLOB graphs
mapper_name = 'mapper'
mapper_columns : str = (f'CREATE TABLE {database}.{mapper_name} ('
#                            "id UInt64, "
                            "File UInt64, "
                            'ModelID UInt32, '
                            'Profile String, '
                            'Attempt UInt32, '
                            # 'Plots String'
                            'Plots Tuple('
                                'train String, '
                                'valid String'
                            ') '
                        ') '
                        'ENGINE = MergeTree() '
                        'ORDER BY (File, ModelID, Attempt);'
                )
# print(histories_columns)
client.execute(mapper_columns)



# %%
client.execute(show_tables())
client.execute(describe_table(mapper_name))
client.execute(get_table(mapper_name))
#client.execute(drop_table(mapper_name))
# %%
#? Fill Table 1
data = pd.read_csv('Models/history.csv')
print(data.columns[:])

file_N = 1
data = data.drop(['loss', 'train_l', 'vall_l', 'test_l'], axis=1)
data = data.rename(columns={'time(s)' : 'time_s', 
                     })
data['File'] = file_N

data['train'] = list(zip(data['train_mae'], data['train_rms'], data['train_r_s']))
data = data.drop(['train_mae', 'train_rms', 'train_r_s'], axis=1)
data['val'] = list(zip(data['val_mae'], data['val_rms'], data['val_r_s'], data['val_t_s']))
data = data.drop(['val_mae', 'val_rms', 'val_r_s', 'val_t_s'], axis=1)
data['tes'] = list(zip(data['tes_mae'], data['tes_rms'], data['tes_r_s'], data['tes_t_s']))
data = data.drop(['tes_mae', 'tes_rms', 'tes_r_s', 'tes_t_s'], axis=1)

#! Decide what to do with NaNs
data = data.dropna()

# data.columns = ((c, ) for c in data.columns)
# INSERT INTO ml_base.histories (File, Epoch, mae, rmse, rsquare, time_s, train) VALUES( 0, 0.1, 0.2, 0.3 ,0.4, 0.5, (1.6, 1.6, 1.6) )
order = ['File', 'Epoch', 'mae', 'rmse', 'rsquare', 'time_s', 'learn_r', 'train', 'val', 'tes']
order_str = ', '.join(order)
query = (f'INSERT INTO {database}.{histores_name} ({order_str}) VALUES')
for i in range(len(data)):
    cut = ', '.join((str(v) for v in data.loc[i,order].values))
    query += f'({cut}), '
    # query = (f'INSERT INTO {database}.{histores_name} ({order_str}) VALUES ({cut});')
query = query[:-2] + ';' # Cul the last space and comma and close query with ;
client.execute(query=query)
# %%
#? Fill table 2
order = ['File', 'ModelID', 'Profile', 'Attempt', 'Plots'] #! BEst Plots for report, Name of the Author
order_str = ', '.join(order)
with open('figures/accuracy.svg', 'rb') as file:
    train = file.read().decode('utf-8')
with open('figures/plot-example.svg', 'rb') as file:
    valid = file.read().decode('utf-8')


query = (f'INSERT INTO {database}.{mapper_name} ({order_str}) VALUES ')
query += f"({file_N}, {1}, 'DST', {1}, ('{train}', '{valid}'));"

client.execute(query=query)

# plot_back = client.execute(query=f'SELECT Plots.train FROM {database}.{mapper_name}')
# with open('figures/plot-example-CH.svg', 'wb') as file:
#     file.write(plot_back[0][0].encode('utf-8'))

# thedata = open('thefile', 'rb').read()
# sql = "INSERT INTO sometable (theblobcolumn) VALUES (%s)"
# cursor.execute(sql, (thedata,))

# with open('query.sql', 'w') as file:
#     file.write(query)


# %%
# client.execute(
#     'INSERT INTO ml_base.histories (File, Epoch, mae, rmse, rsquare, time_s, train) VALUES'
#          '(0, 1, 0.2, 0.3 ,0.4, 0.5, (1.6, 1.6, 1.6)),'
#          '(1, 2, 0.2, 0.3 ,0.4, 0.5, (1.7, 1.7, 1.7))'
#          ';'
#     )
# client.insert_dataframe(query=f'INSERT INTO {database}.{histores_name} ({order_str}) VALUES',
#                         dataframe=data[order], settings={'use_numpy': True})

# l = [('train', 'mae'), ('train', 'rms'), ('train', 'r_s')]
# df = pd.DataFrame(data=np.array([data['train.mae'].to_numpy(),
#                    data['train.rms'].to_numpy(),
#                    data['train.r_s'].to_numpy()]).T, columns=l)
# df.columns = pd.MultiIndex.from_tuples(df.columns)
# pd.concat([data[['File', 'Epoch', 'mae', 'rmse', 'rsquare', 'time_s']], df])
# %%
client.execute(f'INSERT INTO {database}.{histores_name} VALUES'
                '(0, 0.1, 0.2, 0.3 ,0.4, 0.5, (1.6, 1.6 ,1.6))')
# %%
data2 = pd.MultiIndex.from_frame(data)


query=f'INSERT INTO {database}.{histores_name} VALUES'
with client.disconnect_on_error(
    query=query,
    settings={'use_numpy': True}):
    client.connection.send_query(query, query_id=None)
    client.connection.send_external_tables(None)

    sample_block = client.receive_sample_block()
    columns = [x[0] for x in sample_block.columns_with_types]
    
    client.receive_end_of_query()


def insert_multi_dataframe(
        client : Client, query, dataframe, external_tables=None, query_id=None,
        settings=None):
    """
    *New in version 0.2.0.*

    Inserts pandas DataFrame with specified query.

    :param query: query that will be send to server.
    :param dataframe: pandas DataFrame.
    :param external_tables: external tables to send.
                            Defaults to ``None`` (no external tables).
    :param query_id: the query identifier. If no query id specified
                        ClickHouse server will generate it.
    :param settings: dictionary of query settings.
                        Defaults to ``None`` (no additional settings).
    :return: number of inserted rows.
    """

    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        raise RuntimeError('Extras for NumPy must be installed')

    start_time = time()

    with client.disconnect_on_error(query, settings):
        client.connection.send_query(query, query_id=query_id)
        client.connection.send_external_tables(external_tables)

        sample_block = client.receive_sample_block()
        rv = None
        if sample_block:
            columns_types = [x for x in sample_block.columns_with_types]
            data = [dataframe[column].values for column in columns]
            rv = client.send_data(sample_block, data, columnar=True)
            client.receive_end_of_query()

        client.last_query.store_elapsed(time() - start_time)
        return rv

# %%
data['train', 'mae'] = data[('train.mae',)]
data['train', 'rms'] = data[('train.rms',)]
data['train', 'r_s'] = data[('train.r_s',)]
data = data.drop([('train.mae',), ('train.rms',), ('train.r_s',)], axis=1)
data.columns = pd.MultiIndex.from_tuples(data.columns)


data['val', 'mae'] = data['val.mae']
data['val', 'rms'] = data['val.rms']
data['val', 'r_s'] = data['val.r_s']
data['val', 't_s'] = data['val.t_s']

data = data.drop(['val.mae', 'val.rms', 'val.r_s', 'val.t_s'], axis=1)
data['tes', 'mae'] = data['tes.mae']
data['tes', 'rms'] = data['tes.rms']
data['tes', 'r_s'] = data['tes.r_s']
data['tes', 't_s'] = data['tes.t_s']

data = data.drop(['tes.mae', 'tes.rms', 'tes.r_s', 'tes.t_s'], axis=1)
data = data.reset_index()
data = data.set_index('index')
# %%
# client.execute(
#      f'CREATE TABLE {database}.{histores_name}',
#      [{'Chooser': 'Int64'}, {'Epoch': 'Int32'}, {'mae': 'Float32'}, {'learn_r': 'Float32'}]
# )
