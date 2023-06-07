# %%
import os                       # OS, SYS, argc functions
import pandas as pd             # File read
import matplotlib as mpl        # Plot functionality
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import tensorflow_addons as tfa

sys.path.append(os.getcwd() + '/..')
from py_modules.Attention import *

import platform        # System for deligates, not the platform string
import time

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


profile : str = 'DST'
# Getting Data from excel files.
float_dtype : type = np.float32
valid_dir : str = '../Data/A123_Matt_Val'
columns   : list[str] = [
                        'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                        'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                    ]


def diffSoC(chargeData   : pd.Series,
            discargeData : pd.Series) -> pd.Series:
    """ Return SoC based on differnece of Charge and Discharge Data.
    Data in range of 0 to 1.
    Args:
        chargeData (pd.Series): Charge Data Series
        discargeData (pd.Series): Discharge Data Series

    Raises:
        ValueError: If any of data has negative
        ValueError: If the data trend is negative. (end-beg)<0.

    Returns:
        pd.Series: Ceil data with 2 decimal places only.
    """
    # Raise error
    if((any(chargeData) < 0)
        |(any(discargeData) < 0)):
        raise ValueError("Parser: Charge/Discharge data contains negative.")
    #TODO: Finish up this check
    # if((chargeData[-1] - chargeData[0] < 0)
    #    |(discargeData[-1] - discargeData[0] < 0)):
    #     raise ValueError("Parser: Data trend is negative.")
    return np.round((chargeData - discargeData)*100)/100

def Read_Excel_File(path : str, profile : range,
                    columns : list[str]) -> pd.DataFrame:
    """ Reads Excel File with all parameters. Sheet Name universal, columns,
    type taken from global variables initialization.

    Args:
        path (str): Path to files with os.walk

    Returns:
        pd.DataFrame: Single File frame.
    """
    try:
      df : pd.DataFrame = pd.read_excel(io=path,
                        sheet_name='Channel_1-006',
                        header=0, names=None, index_col=None,
                        usecols=['Step_Index'] + columns,
                        squeeze=False,
                        dtype=float_dtype,
                        engine='openpyxl', converters=None, true_values=None,
                        false_values=None, skiprows=None, nrows=None,
                        na_values=None, keep_default_na=True, na_filter=True,
                        verbose=False, parse_dates=False, date_parser=None,
                        thousands=None, comment=None, skipfooter=0,
                        convert_float=True, mangle_dupe_cols=True
                      )
    except:
      df : pd.DataFrame = pd.read_excel(io=path,
                        sheet_name='Channel_1-005',
                        header=0, names=None, index_col=None,
                        usecols=['Step_Index'] + columns,
                        squeeze=False,
                        dtype=float_dtype,
                        engine='openpyxl', converters=None, true_values=None,
                        false_values=None, skiprows=None, nrows=None,
                        na_values=None, keep_default_na=True, na_filter=True,
                        verbose=False, parse_dates=False, date_parser=None,
                        thousands=None, comment=None, skipfooter=0,
                        convert_float=True, mangle_dupe_cols=True
                      )
    df = df[df['Step_Index'].isin(profile)]
    df = df.reset_index(drop=True)
    df = df.drop(columns=['Step_Index'])
    df = df[columns]   # Order columns in the proper sequence
    return df

for _, _, files in os.walk(valid_dir):
    files.sort(key=lambda f: int(f[-13:-5])) # Sort by last dates
    # Initialize empty structures
    train_X : list[pd.DataFrame] = []
    train_Y : list[pd.DataFrame] = []
    for file in files[0:1]:
        X : pd.DataFrame = Read_Excel_File(valid_dir + '/' + file,
                                    range(22,25), columns) #! or 21
        Y : pd.DataFrame = pd.DataFrame(
                data={'SoC' : diffSoC(
                            chargeData=X.loc[:,'Charge_Capacity(Ah)'],
                            discargeData=X.loc[:,'Discharge_Capacity(Ah)']
                            )},
                dtype=float_dtype
            )
        X = X[['Current(A)', 'Voltage(V)', 'Temperature (C)_1']]
        train_X.append(X)
        train_Y.append(Y)


look_back : int = 32
scaler_MM : MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
scaler_SS : StandardScaler = StandardScaler()
def roundup(x : float, factor : int = 10) -> int:
    """ Round up to a factor. Uses it to create hidden neurons, or Buffer size.
    TODO: Make it a smarter rounder.
    Args:
        x (float): Original float value.
        factor (float): Factor towards which it has to be rounder

    Returns:
        int: Rounded up value based on factor.
    """
    if(factor == 10):
        return int(np.ceil(x / 10)) * 10
    elif(factor == 100):
        return int(np.ceil(x / 100)) * 100
    elif(factor == 1000):
        return int(np.ceil(x / 1000)) * 1000
    else:
        print("Factor of {} not implemented.".format(factor))
        return None


def create_Batch_dataset(X : list[np.ndarray], Y : list[np.ndarray],
                    look_back : int = 1
                    ) -> tuple[np.ndarray, np.ndarray]:
    
    batch : int = len(X)
    dataX : list[np.ndarray] = []
    dataY : list[np.ndarray] = []
    
    for i in range(0, batch):
        d_len : int = X[i].shape[0]-look_back
        dataX.append(np.zeros(shape=(d_len, look_back, 3),
                    dtype=float_dtype))
        dataY.append(np.zeros(shape=(d_len,), dtype=float_dtype))    
        for j in range(0, d_len):
            #dataX[i, j, :, :] = dataset[i:(i+look_back), j:j+1]
            #dataY[i, j]       = dataset[i + look_back, j:j+1]
            dataX[i][j,:,:] = X[i][j:(j+look_back), :]  
            dataY[i][j]     = Y[i][j+look_back,]
    return dataX, dataY

sample_size : int = 0
for i in range(0, len(train_X)):
    # tr_np_X[i,:,:] = scaler_SS.fit_transform(train_X[i][:7095])
    # tr_np_Y[i,:,:] = scaler_MM.fit_transform(train_Y[i][:7095])
    #train_X[i] = scaler_SS.fit_transform(train_X[i])
    train_X[i] = train_X[i].to_numpy()
    train_Y[i] = scaler_MM.fit_transform(train_Y[i])
    sample_size += train_X[i].shape[0]

trX, trY = create_Batch_dataset(train_X, train_Y, look_back)
# %%
#? Model №1 - Chemali2017    - DST  - 45
#?                           - FUDS - 48
#? Model №2 - BinXiao2020    - DST  - 50
#?                           - FUDS - 50
#? Model №3 - TadeleMamo2020 - DST  - 19
#?                           - FUDS - 10
#? Model №5 - GelarehJavid2020 - DST  - 2
#?                             - FUDS - 7
#? Model №6 - WeiZhang2020   - DST  - 9
#?                           - FUDS - 3
authors : str = [ 'Chemali2017', 'BinXiao2020', 'TadeleMamo2020',
                  'GelarehJavid2020', 'WeiZhang2020']
profile : str = 'FUDS'#'d_DST' 'US06' 'FUDS'
versions: str = ['48', '50', '10', '7', '3']

# session  = tf.compat.v1.Session()
# graph = tf.compat.v1.get_default_graph()

# with graph.as_default():
#     with session.as_default():
#         model = tf.keras.models.load_model(model_h5_file, compile=False)
#         model.summary()
#         model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
#             optimizer=tf.optimizers.Adam(learning_rate=0.001,
#                     beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
#             metrics=[tf.metrics.MeanAbsoluteError(),
#                      tf.metrics.RootMeanSquaredError(),
#                      tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
#             )
#         run_meta = tf.compat.v1.RunMetadata()
#         opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

#         flops = tf.compat.v1.profiler.profile(graph=graph,
#                             run_meta=run_meta, cmd='op', options=opts
#                             )
#         print('The v1FLOPs is: {}'.format(flops.total_float_ops), flush=True)
# %%
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
N : int = 4
model_h5_file : str = f'../Models/{authors[N]}/{profile}-models/{versions[N]}'
# lstm_model = tf.keras.models.load_model(model_h5_file, compile=False)
model : tf.keras.models.Sequential = tf.keras.models.load_model(
        model_h5_file,
        compile=False,
        custom_objects={"RSquare": tfa.metrics.RSquare,
                        "AttentionWithContext": AttentionWithContext,
                        "Addition": Addition,
                        }
        )
model.summary()
print('The FLOPs is: {}'.format(get_flops(model)), flush=True)
print(f'For {authors[N]} and {versions[N]}')
# %%