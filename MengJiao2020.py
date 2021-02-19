#!/usr/bin/python
# %% [markdown]
# # # 1
# # #
# # GRU for SoC by Meng Jiao 2020 based momentum optimized algorithm

# In the mmentum gradient method, the current weight change direction takes a
#compromise of the gradient direction at current instant and at historical time
#to prevent the oscilation of weight change and to improve the SoC estimation
#speed.

# Random noise are added to the sampled data, o as to prevent overfitting of the
#GRU model.

# The amprere-hour integration method computes the SoC by integrating the
#current over time [6,7]. It can be easily realized, but this open-loop SoC
#estimation may deviate from the true SoC due to inaccurate SiC initialization.
# The momentum gradient algorithm is proposed to solve oscilation problem
#occured in weight updating if the GRU model during the grafient algorithm
#implementing.

# Takes following vector as input:
# Ψ k = [V (k), I(k)], where V (k), I(k)
# And outputs:
# output SoC as 

# The classic Gradient algorithm in Tensorflow refered as SGD.
# The momentum gradient algorithm takes into account the gradient at current
#instant and at historical time.

# Data Wnt through MinMax normalization. All of them. To eliminate the influence
#of the dimensionality between data. Reduce the network load and improve speed.
# Solve overfitting by adding Random noise??? Chego blin?

# Structure of model = (Input(1,2)-> GRU(units, Stateful) -> Output(1))
# Neuron number - Lowest RMSE 30.

# The cost funtion:
# E = 0.5*(y_true-y_pred)^2     (13)
# Optimizer SGD - Singma of momentum of 0.03-0.10 gives best results it seems.
#0.20 makes overshoot.
# Standard metricks of MAE and RMSE. Additional Custom metric of R2

# R2 = 1 - (sum((y_true-y_pred)^2))/((sum(y_true-y_avg))^2)

# 400 Epochs until RMSE = 0.150 or MAE = 0.0076
# %%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
                        'Current(A)', 'Voltage(V)',
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
        X = X[['Current(A)', 'Voltage(V)']]
        train_X.append(X)
        train_Y.append(Y)
# %%
look_back : int = 1
scaler_MM : MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
scaler_SS : MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
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
        dataX.append(np.zeros(shape=(d_len, look_back, X[i].shape[1]),
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
    train_X[i] = scaler_SS.fit_transform(train_X[i])
    #train_X[i] = train_X[i].to_numpy()
    train_Y[i] = scaler_MM.fit_transform(train_Y[i])
    sample_size += train_X[i].shape[0]

trX, trY = create_Batch_dataset(train_X, train_Y, look_back)
# %%
def custom_loss(y_true : tf.Tensor, y_pred : tf.Tensor) -> tf.Tensor:
    y_pred = tf.convert_to_tensor(value=y_pred)
    y_true = tf.dtypes.cast(x=y_true, dtype=y_pred.dtype)
    return (tf.math.squared_difference(x=y_pred, y=y_true))/2
    
    

h_nodes : int = roundup(sample_size / (6 * ((look_back * 1)+1)))
print(f"The number of hidden nodes is {h_nodes}.")
h_nodes = 30
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(batch_input_shape=(1, look_back, 2)),
    tf.keras.layers.GRU(units=h_nodes, activation='tanh',
        recurrent_activation='sigmoid', use_bias=True,
        kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
        bias_initializer='zeros', kernel_regularizer=None,
        recurrent_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, kernel_constraint=None,
        recurrent_constraint=None, bias_constraint=None, dropout=0,
        recurrent_dropout=0, return_sequences=False, return_state=False,
        go_backwards=False, stateful=True, unroll=False, time_major=False,
        reset_after=True),
    tf.keras.layers.Dense(units=1, activation=None)
])
print(model.summary())
model.compile(loss=custom_loss, optimizer=tf.keras.optimizers.SGD(
                    learning_rate=0.01, momentum=0.3,
                    nesterov=False, name='SGDwM'),
              metrics=[tf.metrics.MeanAbsoluteError(),
                       tf.metrics.RootMeanSquaredError()]
            )
min_rmse = 100
#histories = []
firtstEpoch : bool = True
# %%
history = model.fit(trX[0], trY[0], epochs=1, batch_size=1,
                verbose=1, shuffle=False)
model.reset_states()
plt.plot(model.predict(trX[0], batch_size=1))
model.reset_states()
# %%
epochs : int = 6 #! 37*12 seconds = 444s
file_path = 'Models/Stateful/LSTM_test11_SOC'
for i in range(1,epochs+1):
    print(f'Epoch {i}/{epochs}')
    for i in range(0, len(trX)):
        history = model.fit(trX[i], trY[i], epochs=1, batch_size=1,
                verbose=1, shuffle=False)
        #model.reset_states()    #! Try next time without reset
    # for j in range(0,trX.shape[0]):
    #     model.train_on_batch(trX[j,:,:,:], trY[j,:])
    #! Wont work. Needs a Callback for that.
    # if(i % train_df.shape[0] == 0):
    #     print("Reseting model")
    if(history.history['root_mean_squared_error'][0] < min_rmse):
        min_rmse = history.history['root_mean_squared_error'][0]
        model.save(file_path)

    #histories.append(history)
    
    
    # Saving history variable
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)
    # # or save to csv:
    with open('Models/Stateful/LSTM_test11_SOC-history.csv', mode='a') as f:
        if(firtstEpoch):
            hist_df.to_csv(f, index=False)
            firtstEpoch = False
        else:
            hist_df.to_csv(f, index=False, header=False)
model.save('Models/Stateful/LSTM_test11_SOC_Last')