#!/usr/bin/python
# %% [markdown]
# # Stateful GRU example based on Cosine wave.
# Design will use onlt SoC to predict RUL. This method will also investigate
#batching mechanism for individual files to speed the process.

# Original file: /cosine_Stateful
# %%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from parser.soc_calc import diffSoC
# Configurate GPUs
# Define plot sizes
#! Select GPU for usage. CPU versions ignores it
GPU=0
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    #! With /device/GPU:1 the output was faster.
    #! need to research more why.
    tf.config.experimental.set_visible_devices(
                            physical_devices[GPU], 'GPU')

    #if GPU == 1:
    tf.config.experimental.set_memory_growth(
                            physical_devices[GPU], True)
    print("GPU found and memory growth enabled") 
    
    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    print("GPU found") 
    print(f"\nPhysical GPUs: {len(physical_devices)}"
          f"\nLogical GPUs: {len(logical_devices)}")
#! For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float32')
# %%
# Getting Data from excel files.
float_dtype : type = np.float32
train_dir : str = 'Data/A123_Matt_Set'
valid_dir : str = 'Data/A123_Matt_Val'
columns   : list[str] = ['Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)']

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

#? Getting training data and separated file by batch
for _, _, files in os.walk(train_dir):
    files.sort(key=lambda f: int(f[-13:-5])) # Sort by last dates
    # Initialize empty structures
    train_df : pd.DataFrame = Read_Excel_File(train_dir + '/' + files[0],
                                range(18,25), columns)
    train_df : pd.DataFrame = pd.DataFrame(
            data={'batch_0' : diffSoC(
                        chargeData=(train_df.loc[:,'Charge_Capacity(Ah)']),
                        discargeData=(train_df.loc[:,'Discharge_Capacity(Ah)'])
                        )},
            dtype=float_dtype
        )    
    i : int = 1
    for file in files[1:]:
        df : pd.DataFrame = Read_Excel_File(train_dir + '/' + file,
                                    range(18,25), columns)
        df: pd.DataFrame = pd.DataFrame(
                data={'SoC' : diffSoC(
                            chargeData=df.loc[:,'Charge_Capacity(Ah)'],
                            discargeData=df.loc[:,'Discharge_Capacity(Ah)']
                            )},
                dtype=float_dtype
            )
        train_df[f'batch_{i}'] = df.copy(deep=True)
        i += 1
#? Getting Validation/Testing data in one column
for _, _, files in os.walk(valid_dir):
    files.sort(key=lambda f: int(f[-13:-5])) # Sort by last dates
    # Initialize empty structures
    valid_df : pd.DataFrame = Read_Excel_File(valid_dir + '/' + files[0],
                                range(5,19), columns)
    valid_df : pd.DataFrame = pd.DataFrame(
            data={'SoC' : diffSoC(
                        chargeData=(valid_df.loc[:,'Charge_Capacity(Ah)']),
                        discargeData=(valid_df.loc[:,'Discharge_Capacity(Ah)'])
                        )},
            dtype=float_dtype
        )
    #data_SoC['SoC(%)'] = applyMinMax(data_SoC['SoC'])
    #i : int = 1
    for file in files[1:]:
        df : pd.DataFrame = Read_Excel_File(valid_dir + '/' + file,
                                    range(5,19), columns)
        df: pd.DataFrame = pd.DataFrame(
                data={'SoC' : diffSoC(
                            chargeData=df.loc[:,'Charge_Capacity(Ah)'],
                            discargeData=df.loc[:,'Discharge_Capacity(Ah)']
                            )},
                dtype=float_dtype
            )
        #SoC['SoC(%)'] = applyMinMax(SoC['SoC'])

        # train_df[f'batch_{i}'] = df.copy(deep=True)
        # i += 1
        valid_df = valid_df.append(df.copy(deep=True), ignore_index=True)
# %%
look_back : int = 1
scaler : MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
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

def create_noBatch_dataset(dataset : np.ndarray, look_back : int = 1
                    ) -> tuple[np.ndarray, np.ndarray]:
    
    reshaped : np.ndarray = np.zeros(shape=(dataset.shape[0]*dataset.shape[1]),
                                  dtype=float_dtype)
    for i in range(0, dataset.shape[1]):
        reshaped[i*dataset.shape[0]:i*dataset.shape[0]+dataset.shape[0]] = \
                                                            dataset[:,i]
    
    d_len : int = len(reshaped)-look_back
    dataX : np.ndarray = np.zeros(shape=(d_len, look_back, 1),
                                  dtype=float_dtype)
    dataY : np.ndarray = np.zeros(shape=(d_len),
                                  dtype=float_dtype)
    for i in range(0, d_len):
        dataX[i,:,:] = reshaped[i:(i+look_back)]
        dataY[i]     = reshaped[i + look_back]
    return dataX, dataY

def create_Batch_dataset(dataset : np.ndarray, look_back : int = 1
                    ) -> tuple[np.ndarray, np.ndarray]:
    
    d_len : int = dataset.shape[0]-look_back
    batch : int = dataset.shape[1]

    dataX : np.ndarray = np.zeros(shape=(d_len, batch, look_back, 1),
                                  dtype=float_dtype)
    dataY : np.ndarray = np.zeros(shape=(d_len, batch),
                                  dtype=float_dtype)
    for i in range(0, d_len):
        for j in range(0, batch):
            dataX[i, j, :, :] = dataset[i:(i+look_back), j]
            dataY[i, j]       = dataset[i + look_back, j]
    return dataX, dataY

train_df = scaler.fit_transform(train_df)
valid_df = scaler.fit_transform(valid_df)
# trX, trY = create_noBatch_dataset(train_df, look_back)
tsX, tsY = create_noBatch_dataset(valid_df, look_back)

trX, trY = create_Batch_dataset(train_df, look_back)

print("Data Shapes:\n"
      "Train {}\n"
      "TrainX {}\n"
      "TrainY {}\n".format(train_df.shape,trX.shape, trY.shape)
      )
# %%
h_nodes : int = roundup(len(trX) / (6 * ((look_back * 1)+1)))
print(f"The number of hidden nodes is {h_nodes}.")
h_nodes = 64
batch_size = train_df.shape[1]
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(batch_input_shape=(batch_size, look_back, 1)),
    tf.keras.layers.GRU(units=h_nodes, stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.GRU(units=h_nodes, stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.GRU(units=int(h_nodes/2), stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
    tf.keras.layers.GRU(units=int(h_nodes/2), stateful=True, return_sequences=False),
    tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=1, activation=None)
])
print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=[tf.metrics.MeanAbsoluteError(),
                       tf.metrics.RootMeanSquaredError()]
            )
# %%
epochs : int = 250
for i in range(1,epochs+1):
    print(f'Epoch {i}/{epochs}')
    # model.fit(trX[:,:,:,:], trY[:,:], epochs=1, batch_size=12,
    #             verbose=1, shuffle=False)
    for j in range(0,trX.shape[0]):
        model.train_on_batch(trX[j,:,:,:], trY[j,:])
    #! Wont work. Needs a Callback for that.
    # if(i % train_df.shape[0] == 0):
    #     print("Reseting model")
    model.reset_states()    
# %%
model.evaluate(trX, trY, batch_size=12, verbose=1)
new_model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(batch_input_shape=(1, look_back, 1)),
    tf.keras.layers.GRU(units=h_nodes, stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.GRU(units=h_nodes, stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.GRU(units=int(h_nodes/2), stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
    tf.keras.layers.GRU(units=int(h_nodes/2), stateful=True, return_sequences=False),
    tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=1, activation=None)
])
new_model.set_weights(model.get_weights())
new_model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=[tf.metrics.MeanAbsoluteError(),
                       tf.metrics.RootMeanSquaredError()]
            )
# %%
new_model.evaluate(tsX[:252], tsY[:252], 
                            batch_size=1, verbose=1)
new_model.reset_states()

new_model.evaluate(trX[:,0,:,:], trY[:,0], batch_size=1, verbose=1)

look_ahead : int = 250
#trainPredict = [np.vstack([trX[-1][1:], trY[-1]])]
trainPredict = [np.vstack([trX[-1,0,:,:], trY[-1,0]])]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = new_model.predict(np.array([trainPredict[-1]]),
                                batch_size=batch_size)
    predictions[i] = prediction
    trainPredict.append(np.vstack([trainPredict[-1][1:],prediction]))

plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
# plt.plot(np.arange(look_ahead),dataset[train_size:(train_size+look_ahead)],
#             label="test function")
plt.legend()
plt.show()
# %%