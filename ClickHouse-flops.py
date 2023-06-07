# %%
import tempfile
import sys

from clickhouse_driver import Client

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from py_modules.Attention import *
from pypapi import events, papi_high as high

physical_devices = tf.config.experimental.list_physical_devices('GPU')

# %%
database = 'ml_base'
client = Client(host='192.168.1.254', port=9000, 
                database=database, user='tf', password='TF28')

client.execute('SHOW DATABASES')
# %%
def show_tables():
    return client.execute(f'SHOW TABLES FROM {database};')
def describe_table(table : str):
    return client.execute(f'DESCRIBE TABLE {database}.{table} SETTINGS describe_include_subcolumns=1;')
def describe_simple_table(table : str):
    return client.execute(f'DESCRIBE TABLE {database}.{table};')
def get_table(table : str):
    return client.execute(f'SELECT * FROM {database}.{table}')

#? 1) Mapper table ['File', 'ModelID', 'Profile', 'Attempt', 'Name' 'Hist_plots(train, valid)',
#?                                                    'Best_plots(train, valid, test1, test2)']
#? 2) Histories table ['File', 'Epoch', 'mae', 'rmse', 'rsquare', 'time_s', 'learn_r',
#?                                                                       'train()', 'val()', 'tes()']
#? 3) Faulty Histories ['File', 'Epoch', 'attempt', 'mae', 'time_s', learn_rate, train ()]
#? 4) Logits table logits_(train, valid,test) ['File', 'Epoch', logits()]
#? 5) Models table ['File', 'Epochs', 'Latest_Model', 'TR_Model', 'VL_Model', 'TS_Model']
#? 6) Full train logits table ['File', 'Epochs', 'Logits']
mapper : str = 'mapper'
histores : str = 'histories'
faulties : str = 'faulties'
models : str = 'models'
accuracies : str = 'logits_full_train'
def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
            [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs]    
        )
    _, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta,
                cmd='op', options=opts
            )
        return flops.total_float_ops
# %%
# tmpfile = tempfile.SpooledTemporaryFile(max_size=500000, mode='wb',suffix='.h5')
with tempfile.NamedTemporaryFile(mode='wb',suffix='.h5') as file:
    file.write(
            client.execute(
                    f"SELECT latest FROM {database}.models WHERE id = ("
                    f"SELECT id FROM {database}.mapper WHERE (ModelID = 1 AND Attempt = 2 AND Profile = 'US06' )"
                    ");"
                )[0][0]
        )
    
    model : tf.keras.models.Sequential = tf.keras.models.load_model(
            file.name,
            custom_objects={
                        "AttentionWithContext": AttentionWithContext,
                        "Addition": Addition,
                        },
            compile=True
        )
    model.summary()
    #high.start_counters([events.PAPI_FP_OPS,])
    # Do something
    print('The FLOPs is: {}'.format(get_flops(model)), flush=True)
    #x=high.stop_counters()
# %%