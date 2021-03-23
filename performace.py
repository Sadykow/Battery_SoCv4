#!/usr/bin/python
# %% [markdown]
# TFLite converter and performance measurer
# 
# %%
import os                       # OS, SYS, argc functions
import pandas as pd             # File read
import matplotlib as mpl        # Plot functionality
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tflite_runtime.interpreter as tflite

import platform        # System for deligates, not the platform string
import time

# %%
# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# Set Delegates
EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

# %%
profile : str = 'DST'
# Getting Data from excel files.
float_dtype : type = np.float32
valid_dir : str = 'Data/A123_Matt_Val'
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

#? Getting Validation/Testing data in one column
# for _, _, files in os.walk(valid_dir):
#     files.sort(key=lambda f: int(f[-13:-5])) # Sort by last dates
#     # Initialize empty structures
#     valid_df : pd.DataFrame = Read_Excel_File(valid_dir + '/' + files[0],
#                                 range(5,19), columns)
#     valid_df : pd.DataFrame = pd.DataFrame(
#             data={'SoC' : diffSoC(
#                         chargeData=(valid_df.loc[:,'Charge_Capacity(Ah)']),
#                         discargeData=(valid_df.loc[:,'Discharge_Capacity(Ah)'])
#                         )},
#             dtype=float_dtype
#         )
#     #data_SoC['SoC(%)'] = applyMinMax(data_SoC['SoC'])
#     #i : int = 1
#     for file in files[1:]:
#         df : pd.DataFrame = Read_Excel_File(valid_dir + '/' + file,
#                                     range(5,19), columns)
#         df: pd.DataFrame = pd.DataFrame(
#                 data={'SoC' : diffSoC(
#                             chargeData=df.loc[:,'Charge_Capacity(Ah)'],
#                             discargeData=df.loc[:,'Discharge_Capacity(Ah)']
#                             )},
#                 dtype=float_dtype
#             )
#         #SoC['SoC(%)'] = applyMinMax(SoC['SoC'])

#         # train_df[f'batch_{i}'] = df.copy(deep=True)
#         # i += 1
#         valid_df = valid_df.append(df.copy(deep=True), ignore_index=True)
#? Getting training data and separated file by batch
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
# %%
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
author : str = 'WeiZhang2020'#'BinXiao2020' 'TadeleMamo2020' 'Chemali2017'
profile: str = 'DST'#'d_DST' 'US06' 'FUDS'
#version: str = '39'
model_file : str = f'Models/{author}/{profile}-models/{profile}.tflite'
model_file, *device = model_file.split('@')
interpreter = tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# %%
print('----INFERENCE TIME----')
print('Note: The first inference on Edge TPU is slow because it includes',
    'loading the model into Edge TPU memory.')
for _ in range(5):
    # Test the mode
    input_shape = input_details[0]['shape']
    output_data : np.ndarray = np.zeros(shape=(trX[0].shape[0],))
    start = time.perf_counter()
    for i in range(0, 1):
        interpreter.set_tensor(input_details[0]['index'], trX[0][i:i+1,:,:])   
        interpreter.invoke()
        #output_data[i] = interpreter.get_tensor(output_details[0]['index'])
        interpreter.get_tensor(output_details[0]['index'])
    print('%.1fms' % ((time.perf_counter() - start) * 1000))
    #print('Time took: {}'.format(time.perf_counter() - start))
    
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.    
    print(output_data[i:i+1])

