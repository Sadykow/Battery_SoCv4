#!/usr/bin/python
# %% [markdown]
# # Stateful LSTM example based on Cosine wave.
# Design will use onlt SoC to predict RUL. This method will also investigate
#batching mechanism for individual files to speed the process.

# Original file: /cosine_Stateful
# %%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import dropout

# Configurate GPUs
# Define plot sizes
#! Select GPU for usage. CPU versions ignores it
GPU=1
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
columns   : list[str] = [
                        'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                        'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                    ]

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
    return np.round((chargeData - discargeData)*100)/100

#? Getting training data and separated file by batch
for _, _, files in os.walk(train_dir):
    files.sort(key=lambda f: int(f[-13:-5])) # Sort by last dates
    # Initialize empty structures
    train_X : list[pd.DataFrame] = []
    train_Y : list[pd.DataFrame] = []
    for file in files[0:1]:
        X : pd.DataFrame = Read_Excel_File(train_dir + '/' + file,
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
look_back : int = 1
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

# def create_Batch_dataset(dataX : np.ndarray, dataY : np.ndarray,
#                     look_back : int = 1) -> tuple[np.ndarray, np.ndarray]:
    
#     d_len : int = dataX.shape[1]-look_back
#     batch : int = dataX.shape[0]

#     dX : np.ndarray = np.zeros(shape=(d_len, batch, look_back, dataX.shape[2]),
#                                   dtype=float_dtype)
#     dY : np.ndarray = np.zeros(shape=(d_len, batch),
#                                   dtype=float_dtype)
#     for i in range(0, d_len):
#         for j in range(0, batch):
#             dX[i, j, :, :] = dataX[j, i:(i+look_back), :]
#             dY[i, j]       = dataY[j, i + look_back]
#     return dX, dY
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

# tr_np_X : np.ndarray = np.zeros(shape=(len(train_X), train_X[0].shape[0],
#                                         train_X[0].shape[1]),
#                                 dtype=np.float32, order='C')
# tr_np_Y : np.ndarray = np.zeros(shape=(len(train_Y), train_Y[0].shape[0],
#                                         train_Y[0].shape[1]),
#                                 dtype=np.float32, order='C')
#! Tvoy mat'... All this time, samples were not equal
sample_size : int = 0
for i in range(0, len(train_X)):
    # tr_np_X[i,:,:] = scaler_SS.fit_transform(train_X[i][:7095])
    # tr_np_Y[i,:,:] = scaler_MM.fit_transform(train_Y[i][:7095])
    #train_X[i] = scaler_SS.fit_transform(train_X[i])
    train_X[i] = train_X[i].to_numpy()
    train_Y[i] = scaler_MM.fit_transform(train_Y[i])
    sample_size += train_X[i].shape[0]
# trX, trY = create_noBatch_dataset(train_df, look_back)
#tsX, tsY = create_noBatch_dataset(valid_df, look_back)

trX, trY = create_Batch_dataset(train_X, train_Y, look_back)

# trX = np.reshape(trX[:],newshape=(trX.shape[0]*trX.shape[1],trX.shape[2],trX.shape[3]),order='C')
# trY = np.reshape(trY[:],newshape=(trY.shape[0]*trY.shape[1],)  ,order='C')
# print("Data Shapes:\n"
#       "Train {}\n"
#       "TrainX {}\n"
#       "TrainY {}\n".format(tr_np_X.shape,trX.shape, trY.shape)
    #   )
# %%
#! Feature count missing
h_nodes : int = roundup(sample_size / (6 * ((look_back * 3)+1)))
print(f"The number of hidden nodes is {h_nodes}.")
h_nodes : int = 96
#batch_size = tr_np_X.shape[0]
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(batch_input_shape=(1, look_back, 3)),
    tf.keras.layers.LSTM(units=h_nodes, stateful=True,
                         return_sequences=True, dropout=0.2),
    tf.keras.layers.LSTM(units=h_nodes, stateful=True,
                         return_sequences=True, dropout=0.2),
    tf.keras.layers.LSTM(units=h_nodes, stateful=True,
                         return_sequences=True, dropout=0.2),
    tf.keras.layers.LSTM(units=h_nodes, stateful=True,
                         return_sequences=True, dropout=0.2),
    tf.keras.layers.LSTM(units=h_nodes, stateful=True,
                         return_sequences=True, dropout=0.2),
    tf.keras.layers.LSTM(units=h_nodes, stateful=True,
                         return_sequences=True, dropout=0.2),
    tf.keras.layers.LSTM(units=h_nodes, stateful=True,
                         return_sequences=False, dropout=0.2),
    tf.keras.layers.Dense(units=1, activation=None)
])
print(model.summary())
tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(targets, outputs))))
tf.metrics.MeanAbsolutePercentageError()
model.compile(loss=tf.losses.MeanAbsoluteError(),
              optimizer='adam',
            #   optimizer=tf.optimizers.Adam(learning_rate=10e-04,
            #         beta_1=0.9, beta_2=0.999, epsilon=10e-08,),
              metrics=[tf.metrics.MeanAbsoluteError(),
                       tf.metrics.RootMeanSquaredError()]
            )
min_rmse = 100
#histories = []
firtstEpoch : bool = True
# %%
epochs : int = 30 #! 120*12 second = 1440s
file_path = 'Models/Stateful/LSTM_test10_VIT'
for i in range(1,epochs+1):
    print(f'Epoch {i}/{epochs}')
    for i in range(0, len(trX)):
        history = model.fit(trX[i], trY[i], epochs=1, batch_size=1,
                verbose=1, shuffle=False)
        model.reset_states()    #! Try next time without reset
    if(history.history['root_mean_squared_error'][0] < min_rmse):
        min_rmse = history.history['root_mean_squared_error'][0]
        model.save(file_path)
    #histories.append(history)
    # Saving history variable
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)
    # or save to csv:
    with open('Models/Stateful/LSTM_test10_VIT-history.csv', mode='a') as f:
        if(firtstEpoch):
            hist_df.to_csv(f, index=False)
            firtstEpoch = False
        else:
            hist_df.to_csv(f, index=False, header=False)
model.save('Models/Stateful/LSTM_test10_VIT_Last')
# # %%
# # for j in range(0,trX.shape[0]):
# #     model.evaluate(trX[j,:,:,:], trY[j,:], batch_size=12, verbose=1)
# new_model = tf.keras.models.Sequential([
#     tf.keras.layers.InputLayer(batch_input_shape=(1, look_back, tr_np_X.shape[2])),
#     tf.keras.layers.LSTM(units=h_nodes, stateful=True, return_sequences=True),
#     tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
#     tf.keras.layers.LSTM(units=h_nodes, stateful=True, return_sequences=True),
#     tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
#     tf.keras.layers.LSTM(units=int(h_nodes/2), stateful=True, return_sequences=True),
#     tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
#     tf.keras.layers.LSTM(units=int(h_nodes/2), stateful=True, return_sequences=False),
#     tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
#     tf.keras.layers.Dense(units=1, activation=None)
# ])
# new_model.set_weights(tf.keras.models.load_model(file_path).get_weights())
# new_model.compile(loss='mean_squared_error', optimizer='adam',
#               metrics=[tf.metrics.MeanAbsoluteError(),
#                        tf.metrics.RootMeanSquaredError()]
#             )
# # %%
# new_model.evaluate(tsX[:252], tsY[:252], 
#                             batch_size=1, verbose=1)
# new_model.reset_states()

# new_model.evaluate(trX[::batch_size,:,:], trY[::batch_size], batch_size=1, verbose=1)
# new_model.reset_states()
# model.evaluate(trX, trY, batch_size=batch_size,verbose=1)
# model.reset_states()

# look_ahead = 5000
# pre_train = 2000
# xval = trX[:pre_train*batch_size:batch_size,:,:]
# predictions = np.zeros((pre_train + look_ahead,1))
# predictions[:pre_train] = new_model.predict(xval, batch_size=1)

# for i in range(look_ahead):
#     prediction = new_model.predict(xval[-1:,:,:], batch_size=1)
#     predictions[i] = prediction
#     xval = np.expand_dims(prediction, axis=1)
# new_model.reset_states()

# plt.figure(figsize=(12,5))
# plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
# for i in range(12):
#     plt.plot(trY[i::batch_size],'--')
# plt.legend()
# plt.show()
# # %%
# rmse = np.zeros(shape=(len(histories)) )
# for i in range(0, len(histories)):
#     rmse[i] = (histories[i].history['root_mean_squared_error'])[0]
# plt.plot(rmse)
# %%
#? Single stright line fit over SoC. Always starts from 1 to 0.
#?Try improve with incresed neurons number
# model = tf.keras.models.load_model('Models/Stateful/LSTM_test10_VIT')

#? Or maybe I should not have. Since it makes perfect sense.
# pred = model.predict(trX[1], batch_size=1)
# #pred1 = model.predict(trX[1], batch_size=1)
# #pred2 = model.predict(trX[2], batch_size=1)
# model.reset_states()

# # %%
# 1/(pred[0]-pred[-1])


# total_cycle : int = 8000
# pre_predict : int = 1000
# index_file  : int = 3
# look_ahead = total_cycle - pre_predict
# predictions = np.zeros((look_ahead,1))
# prediction = model.predict(trX[index_file][:pre_predict,:,:], batch_size=1)[-1:,:]
# for i in range(look_ahead):
#     prediction = model.predict(np.expand_dims(prediction,
#                                                axis=1),
#                             batch_size=1)
#     predictions[i] = prediction    
# model.reset_states()

# time = np.linspace(0,trX[index_file].shape[0],8000)
# multiplier : float = trX[index_file][pre_predict:total_cycle,0,0][0]/(predictions[0][0]+0.08)
# plt.figure(figsize=(12,5))
# plt.plot(time[0:pre_predict], trX[index_file][:pre_predict,0,0], 'b-', label='True')
# #plt.plot(time[pre_predict:],(predictions+0.075)*8.5,'r',label="prediction")
# plt.plot(time[pre_predict:],(predictions+0.08)*multiplier,'r',label="prediction")
# plt.plot(time[pre_predict:], trX[index_file][pre_predict:total_cycle,0,0], 'b--', label='True')
# plt.legend()
# plt.title("Remaining Useful Life prediction")
# plt.show()