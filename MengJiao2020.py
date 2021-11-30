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
import datetime
import logging
import os, sys, getopt    # OS, SYS, argc functions
from sys import platform  # Get type of OS

import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # File read
import tensorflow as tf  # Tensorflow and Numpy replacement
import tensorflow_addons as tfa

from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator
from cy_modules.utils import str2bool
from py_modules.plotting import predicting_plot

#from sklearn.preprocessing import MinMaxScaler, StandardScaler
# %%
# Extract params
# try:
#     opts, args = getopt.getopt(sys.argv[1:],"hd:e:g:p:",
#                     ["help", "debug=", "epochs=",
#                      "gpu=", "profile="])
# except getopt.error as err: 
#     # output error, and return with an error code 
#     print (str(err)) 
#     print ('EXEPTION: Arguments requied!')
#     sys.exit(2)

opts = [('-d', 'False'), ('-e', '50'), ('-g', '1'), ('-p', 'd_FUDS')]
mEpoch  : int = 10
GPU     : int = 0
profile : str = 'DST'
for opt, arg in opts:
    if opt == '-h':
        print('HELP: Use following default example.')
        print('python *.py --debug False --epochs 50 --gpu 0 --profile DST')
        print('TODO: Create a proper help')
        sys.exit()
    elif opt in ("-d", "--debug"):
        if(str2bool(arg)):
            logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s --> %(levelname)s:%(message)s')
            logging.warning("Logger DEBUG")
        else:
            logging.basicConfig(level=logging.CRITICAL)
            logging.warning("Logger Critical")
    elif opt in ("-e", "--epochs"):
        mEpoch = int(arg)
    elif opt in ("-g", "--gpu"):
        #! Another alternative is to use
        #!:$ export CUDA_VISIBLE_DEVICES=0,1 && python *.py
        GPU = int(arg)
    elif opt in ("-p", "--profile"):
        profile = (arg)
# %%
# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['font.family'] = 'Bender'

# Configurage logger and print basics
logging.basicConfig(level=logging.CRITICAL,        
    format='%(asctime)s --> %(levelname)s:%(message)s')
logging.warning("Logger enabled")

logging.debug("\n\n"
    f"MatPlotLib version: {mpl.__version__}\n"
    f"Pandas     version: {pd.__version__}\n"
    f"Tensorflow version: {tf.version.VERSION}\n"
    )
logging.debug("\n\n"
    f"Plot figure size set to {mpl.rcParams['figure.figsize']}\n"
    f"Axes grid: {mpl.rcParams['axes.grid']}"
    )
#! Select GPU for usage. CPU versions ignores it.
#!! Learn to check if GPU is occupied or not.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    #! With /device/GPU:1 the output was faster.
    #! need to research more why.
    tf.config.experimental.set_visible_devices(
                            physical_devices[GPU], 'GPU')

    #if GPU == 1:
    tf.config.experimental.set_memory_growth(
                            physical_devices[GPU], True)
    logging.info("GPU found and memory growth enabled") 
    
    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    logging.info("GPU found") 
    logging.debug(f"\nPhysical GPUs: {len(physical_devices)}"
                  f"\nLogical GPUs: {len(logical_devices)}")
#! For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float32')
# %%
# Getting Data from excel files.
# float_dtype : type = np.float32
# train_dir : str = 'Data/A123_Matt_Set'
# valid_dir : str = 'Data/A123_Matt_Val'
# columns   : list[str] = [
#                         'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
#                         'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
#                     ]

# def Read_Excel_File(path : str, profile : range,
#                     columns : list[str]) -> pd.DataFrame:
#     """ Reads Excel File with all parameters. Sheet Name universal, columns,
#     type taken from global variables initialization.

#     Args:
#         path (str): Path to files with os.walk

#     Returns:
#         pd.DataFrame: Single File frame.
#     """
#     try:
#       df : pd.DataFrame = pd.read_excel(io=path,
#                         sheet_name='Channel_1-006',
#                         header=0, names=None, index_col=None,
#                         usecols=['Step_Index'] + columns,
#                         squeeze=False,
#                         dtype=float_dtype,
#                         engine='openpyxl', converters=None, true_values=None,
#                         false_values=None, skiprows=None, nrows=None,
#                         na_values=None, keep_default_na=True, na_filter=True,
#                         verbose=False, parse_dates=False, date_parser=None,
#                         thousands=None, comment=None, skipfooter=0,
#                         convert_float=True, mangle_dupe_cols=True
#                       )
#     except:
#       df : pd.DataFrame = pd.read_excel(io=path,
#                         sheet_name='Channel_1-005',
#                         header=0, names=None, index_col=None,
#                         usecols=['Step_Index'] + columns,
#                         squeeze=False,
#                         dtype=float_dtype,
#                         engine='openpyxl', converters=None, true_values=None,
#                         false_values=None, skiprows=None, nrows=None,
#                         na_values=None, keep_default_na=True, na_filter=True,
#                         verbose=False, parse_dates=False, date_parser=None,
#                         thousands=None, comment=None, skipfooter=0,
#                         convert_float=True, mangle_dupe_cols=True
#                       )
#     df = df[df['Step_Index'].isin(profile)]
#     df = df.reset_index(drop=True)
#     df = df.drop(columns=['Step_Index'])
#     df = df[columns]   # Order columns in the proper sequence
#     return df

# def diffSoC(chargeData   : pd.Series,
#             discargeData : pd.Series) -> pd.Series:
#     """ Return SoC based on differnece of Charge and Discharge Data.
#     Data in range of 0 to 1.
#     Args:
#         chargeData (pd.Series): Charge Data Series
#         discargeData (pd.Series): Discharge Data Series

#     Raises:
#         ValueError: If any of data has negative
#         ValueError: If the data trend is negative. (end-beg)<0.

#     Returns:
#         pd.Series: Ceil data with 2 decimal places only.
#     """
#     # Raise error
#     if((any(chargeData) < 0)
#         |(any(discargeData) < 0)):
#         raise ValueError("Parser: Charge/Discharge data contains negative.")
#     return np.round((chargeData - discargeData)*100)/100

# #? Getting training data and separated file by batch
# for _, _, files in os.walk(train_dir):
#     files.sort(key=lambda f: int(f[-13:-5])) # Sort by last dates
#     # Initialize empty structures
#     train_X : list[pd.DataFrame] = []
#     train_Y : list[pd.DataFrame] = []
#     for file in files[:]:
#         X : pd.DataFrame = Read_Excel_File(train_dir + '/' + file,
#                                     range(22,25), columns) #! or 21
#         Y : pd.DataFrame = pd.DataFrame(
#                 data={'SoC' : diffSoC(
#                             chargeData=X.loc[:,'Charge_Capacity(Ah)'],
#                             discargeData=X.loc[:,'Discharge_Capacity(Ah)']
#                             )},
#                 dtype=float_dtype
#             )
#         X = X[['Current(A)', 'Voltage(V)']]
#         train_X.append(X)
#         train_Y.append(Y)
# # %%
# look_back : int = 1
# scaler_MM : MinMaxScaler    = MinMaxScaler(feature_range=(0, 1))
# scaler_CC : MinMaxScaler    = MinMaxScaler(feature_range=(0, 1))
# scaler_VV : StandardScaler  = StandardScaler()
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

# def create_Batch_dataset(X : list[np.ndarray], Y : list[np.ndarray],
#                     look_back : int = 1
#                     ) -> tuple[np.ndarray, np.ndarray]:
    
#     batch : int = len(X)
#     dataX : list[np.ndarray] = []
#     dataY : list[np.ndarray] = []
    
#     for i in range(0, batch):
#         d_len : int = X[i].shape[0]-look_back
#         dataX.append(np.zeros(shape=(d_len, look_back, X[i].shape[1]),
#                     dtype=float_dtype))
#         dataY.append(np.zeros(shape=(d_len,), dtype=float_dtype))    
#         for j in range(0, d_len):
#             #dataX[i, j, :, :] = dataset[i:(i+look_back), j:j+1]
#             #dataY[i, j]       = dataset[i + look_back, j:j+1]
#             dataX[i][j,:,:] = X[i][j:(j+look_back), :]  
#             dataY[i][j]     = Y[i][j+look_back,]
#     return dataX, dataY

# sample_size : int = 0
# for i in range(0, len(train_X)):
#     #! Scale better with STD on voltage
#     #train_X[i].iloc[:,0] = scaler_CC.fit_transform(np.expand_dims(train_X[i]['Current(A)'], axis=1))
#     #train_X[i].iloc[:,1] = scaler_VV.fit_transform(np.expand_dims(train_X[i]['Voltage(V)'], axis=1))    
#     train_Y[i] = scaler_MM.fit_transform(train_Y[i])
#     train_X[i] = train_X[i].to_numpy()
#     sample_size += train_X[i].shape[0]
    

# trX, trY = create_Batch_dataset(train_X, train_Y, look_back)
# %%
#! Check OS to change SymLink usage
if(platform=='win32'):
    Data    : str = 'DataWin\\'
else:
    Data    : str = 'Data/'
dataGenerator = DataGenerator(train_dir=f'{Data}A123_Matt_Set',
                              valid_dir=f'{Data}A123_Matt_Val',
                              test_dir=f'{Data}A123_Matt_Test',
                              columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                                'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                                ],
                              PROFILE_range = profile)
# %%
look_back : int = 1 
window = WindowGenerator(Data=dataGenerator,
                        input_width=look_back, label_width=1, shift=1,
                        input_columns=['Current(A)', 'Voltage(V)',
                                                'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,
                        includeTarget=False, normaliseLabal=False,
                        shuffleTraining=False)
# Entire Training set
x_train, y_train = window.train

# For validation use same training
x_valid = x_train[2]
y_valid = y_train[2]

# For test dataset take the remaining profiles.
(x_test_one, x_test_two), (y_test_one, y_test_two) = window.valid
# %%
#! Test with golbal
def custom_loss(y_true : tf.Tensor, y_pred : tf.Tensor) -> tf.Tensor:
    y_pred = tf.convert_to_tensor(value=y_pred)
    y_true = tf.dtypes.cast(x=y_true, dtype=y_pred.dtype)        
    #tf.print(y_pred, output_stream='file://pp-temp.txt')
    #tf.print(y_true, output_stream='file://tt-temp.txt')
    return (tf.math.squared_difference(x=y_pred, y=y_true))/2

# sample_size : int = y_train[4].shape[0]
# h_nodes : int = roundup(sample_size / (15 * ((look_back * 1)+1)))
# print(f"The number of hidden nodes is {h_nodes}.")
h_nodes = 60#120-onDST #! 60 Nodes gives a nice results.

file_name : str = os.path.basename(__file__)[:-3]
model_loc : str = f'Models/{file_name}/{profile}-models/'
# %%
accuracies = pd.read_csv(f'{model_loc}history-{profile}.csv')
col_name = 'root_mean_squared_error' # 'mean_absolute_error' 'loss'
iEpoch = accuracies[col_name][accuracies[col_name] == accuracies[col_name].min()].index[0]
print(iEpoch)
# %%
iEpoch = 0
firstLog : bool = True
try:
    for _, _, files in os.walk(model_loc):
        for file in files:
            if file.endswith('.ch'):
                iEpoch = int(os.path.splitext(file)[0])
    
    gru_model : tf.keras.models.Sequential = tf.keras.models.load_model(
            f'{model_loc}{iEpoch}',
            compile=False)
    firstLog = False
    print("Model Identefied. Continue training.")
except OSError as identifier:
    gru_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape=(1, look_back, 3)),
        tf.keras.layers.GRU(units=h_nodes, activation='tanh',
            recurrent_activation='sigmoid', use_bias=True,
            kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
            bias_initializer='zeros', kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0,
            recurrent_dropout=0, return_sequences=True,
            return_state=False,
            go_backwards=False, stateful=True, unroll=False, time_major=False,
            reset_after=True
            ),
        tf.keras.layers.GRU(units=h_nodes, activation='tanh',
            recurrent_activation='sigmoid', use_bias=True,
            kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
            bias_initializer='zeros', kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0,
            recurrent_dropout=0, return_sequences=False,
            return_state=False,
            go_backwards=False, stateful=True, unroll=False, time_major=False,
            reset_after=True
            ),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    firstLog = True
prev_model = tf.keras.models.clone_model(gru_model,
                                    input_tensors=None, clone_function=None)

checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath =model_loc+f'{profile}-checkpoints/checkpoint',
    monitor='val_loss', verbose=0,
    save_best_only=False, save_weights_only=False,
    mode='auto', save_freq='epoch', options=None,
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=model_loc+
            f'tensorboard/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
        histogram_freq=1, write_graph=True, write_images=False,
        update_freq='epoch', profile_batch=2, embeddings_freq=0,
        embeddings_metadata=None
    )

nanTerminate = tf.keras.callbacks.TerminateOnNaN()

gru_model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=
                    tf.keras.optimizers.SGD(
                    learning_rate=0.001, momentum=0.3, #! Put learning rate to 0
                    nesterov=False, name='SGDwM'),
              metrics=[tf.metrics.MeanAbsoluteError(),
                       tf.metrics.RootMeanSquaredError(),
                       tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
            )
prev_model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=
                    tf.keras.optimizers.SGD(
                    learning_rate=0.001, momentum=0.3, #! Put learning rate to 0
                    nesterov=False, name='SGDwM'),
              metrics=[tf.metrics.MeanAbsoluteError(),
                       tf.metrics.RootMeanSquaredError(),
                       tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
            )

# %%
i_attempts : int = 0
n_attempts : int = 3
while iEpoch < mEpoch:
    iEpoch+=1
    print(f"Epoch {iEpoch}/{mEpoch}")
    # with open("pp-temp.txt", "w") as file:
    #     file.write('')
#! Fit one sample - PRedict next one like model() - read all through file output
#!and hope this makes a good simulation of how online learning will go.
    for (x,y) in zip(x_train[:], y_train[:]):
        history = gru_model.fit(x, y, epochs=1,
                        #validation_data=(x_valid, y_valid),
                        callbacks=[nanTerminate],
                        batch_size=1, shuffle=False
                        )
#! In Mamo methods implement Callback to reset model after 500 steps and then 
#!step by one sample for next epoch to capture shift in data. Hell method, but
#!might be more effective that batching 12 together.
    
        gru_model.reset_states()
    PERF = gru_model.evaluate(x=x_valid,
                               y=y_valid,
                               batch_size=1,
                               verbose=0)
    gru_model.reset_states()
    
    # Saving model
    gru_model.save(filepath=f'{model_loc}{iEpoch}',
                overwrite=True, include_optimizer=True,
                save_format='h5', signatures=None, options=None,
                save_traces=True
        )
    prev_model = tf.keras.models.clone_model(gru_model)

    if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
        os.remove(f'{model_loc}{iEpoch-1}.ch')
    os.mknod(f'{model_loc}{iEpoch}.ch')
    
    # Saving history variable
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)
    hist_df['vall_loss'] = PERF[0]
    hist_df['val_mean_absolute_error'] = PERF[1]
    hist_df['val_root_mean_squared_error'] = PERF[2]
    hist_df['val_r_square'] = PERF[3]
    # or save to csv:
    with open(f'{model_loc}history-{profile}.csv', mode='a') as f:
        if(firstLog):
            hist_df.to_csv(f, index=False)
            firstLog = False
        else:
            hist_df.to_csv(f, index=False, header=False)
    
    #! Run the Evaluate function
    #! Replace with tf.metric function.
    PRED = gru_model.predict(x_valid, batch_size=1)
    gru_model.reset_states()
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_valid[::,]-PRED[:,0])))
    PERF = gru_model.evaluate(x=x_valid,
                            y=y_valid,
                            batch_size=1,
                            verbose=0)
    gru_model.reset_states()
    # otherwise the right y-label is slightly clipped
    predicting_plot(profile=profile, file_name='Model №4',
                    model_loc=model_loc,
                    model_type='GRU Train',
                    iEpoch=f'val-{iEpoch}',
                    Y=y_valid,
                    PRED=PRED,
                    RMS=np.expand_dims(RMS, axis=1),
                    val_perf=PERF,
                    TAIL=y_valid.shape[0],
                    save_plot=True,
                    RMS_plot=True) #! Saving memory from high errors.
    if(PERF[-2] <=0.024): # Check thr RMSE
        print("RMS droped around 2.4%. Breaking the training")
        break
# %%
# tr_pred = np.zeros(shape=(7094,))
# i = 0
# with open("pp-temp.txt", "r") as file1:
#     for line in file1.readlines():
#         tr_pred[i] = float(line.split('[[')[1].split(']]\n')[0])
#         i += 1

# plt.plot(tr_pred)
# plt.plot(trY[0])
# %%
# def smooth(y, box_pts: int) -> np.array:
#     """ Smoothing data using numpy convolve. Based on the size of the
#     averaging box, data gets smoothed.
#     Here it used in following form:
#     y = V/(maxV-minV)
#     box_pts = 500

#     Args:
#         y (pd.Series): A data which requires to be soothed.
#         box_pts (int): Number of points to move averaging box

#     Returns:
#         np.array: Smoothed data array
#     """
#     box = np.ones(box_pts)/box_pts
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth
# # 5 seconds timestep
# plt.figure()
# plt.plot(train_X[0][:,1])
# plt.plot(smooth(train_X[0][:,1], 150))
# #plt.xlim([0, 7000])
# plt.ylim([2.5, 3.7])
# plt.xlabel('TimeSteps (s)')
# plt.ylabel('Voltage (V)')
# plt.grid()
# plt.show()
# %%
# Plot
# import seaborn as sns
# train_X[0]['Time (s)'] = np.linspace(0,7095*5,7095)
# g = sns.relplot(x='Time (s)', y='Temperature (C)_1', kind="line",
#                 data=train_X[0], size=11, color='k')
# #plt.xlim(-100, 40000)
# #plt.ylim(2.25, 3.75)
# g.fig.autofmt_xdate()
# fir = g.fig
# fir.savefig('../1-Voltage.svg', transparent=True)
# # tr_pred = np.zeros(shape=(7094,))
# i = 0
# with open("tt-temp.txt", "r") as file1:
#     for line in file1.readlines():
#         tr_pred[i] = float(line.split('[[')[1].split(']]\n')[0])
#         i += 1
# plt.plot(tr_pred)
# plt.plot(trY[0])
# # %%
# epochs : int = 6 #! 37*12 seconds = 444s
# file_path = 'Models/Stateful/LSTM_test11_SOC'
# for i in range(1,epochs+1):
#     print(f'Epoch {i}/{epochs}')
#     for i in range(0, len(trX)):
#         history = model.fit(trX[i], trY[i], epochs=1, batch_size=1,
#                 verbose=1, shuffle=False)
#         #model.reset_states()    #! Try next time without reset
#     # for j in range(0,trX.shape[0]):
#     #     model.train_on_batch(trX[j,:,:,:], trY[j,:])
#     #! Wont work. Needs a Callback for that.
#     # if(i % train_df.shape[0] == 0):
#     #     print("Reseting model")
#     if(history.history['root_mean_squared_error'][0] < min_rmse):
#         min_rmse = history.history['root_mean_squared_error'][0]
#         model.save(file_path)

#     #histories.append(history)
    
    
#     # Saving history variable
#     # convert the history.history dict to a pandas DataFrame:     
#     hist_df = pd.DataFrame(history.history)
#     # # or save to csv:
#     with open('Models/Stateful/LSTM_test11_SOC-history.csv', mode='a') as f:
#         if(firtstEpoch):
#             hist_df.to_csv(f, index=False)
#             firtstEpoch = False
#         else:
#             hist_df.to_csv(f, index=False, header=False)
# model.save('Models/Stateful/LSTM_test11_SOC_Last')
# %%
PRED = gru_model.predict(x_test_one, batch_size=1, verbose=1)
gru_model.reset_states()
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(y_test_one[::,]-PRED[:,0])))

if profile == 'DST':
    predicting_plot(profile=profile, file_name='Model №4',
                    model_loc=model_loc,
                    model_type='GRU Test on US06', iEpoch=f'Test One-{iEpoch}',
                    Y=y_test_one,
                    PRED=PRED,
                    RMS=np.expand_dims(RMS, 1),
                    val_perf=gru_model.evaluate(
                                    x=x_test_one,
                                    y=y_test_one,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_one.shape[0],
                    save_plot=True,
                    RMS_plot=True)
else:
    predicting_plot(profile=profile, file_name='Model №4',
                    model_loc=model_loc,
                    model_type='GRU Test on DST', iEpoch=f'Test One-{iEpoch}',
                    Y=y_test_one,
                    PRED=PRED,
                    RMS=np.expand_dims(RMS, 1),
                    val_perf=gru_model.evaluate(
                                    x=x_test_one,
                                    y=y_test_one,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_one.shape[0],
                    save_plot=True,
                    RMS_plot=True)
gru_model.reset_states()
PRED = gru_model.predict(x_test_two, batch_size=1, verbose=1)
gru_model.reset_states()
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(y_test_two[::,]-PRED[:,0])))
if profile == 'FUDS':
    predicting_plot(profile=profile, file_name='Model №4',
                    model_loc=model_loc,
                    model_type='GRU Test on US06', iEpoch=f'Test Two-{iEpoch}',
                    Y=y_test_two,
                    PRED=PRED,
                    RMS=np.expand_dims(RMS, 1),
                    val_perf=gru_model.evaluate(
                                    x=x_test_two,
                                    y=y_test_two,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_two.shape[0],
                    save_plot=True,
                    RMS_plot=True)
else:
    predicting_plot(profile=profile, file_name='Model №4',
                    model_loc=model_loc,
                    model_type='GRU Test on FUDS', iEpoch=f'Test Two-{iEpoch}',
                    Y=y_test_two,
                    PRED=PRED,
                    RMS=np.expand_dims(RMS, 1),
                    val_perf=gru_model.evaluate(
                                    x=x_test_two,
                                    y=y_test_two,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_two.shape[0],
                    save_plot=True,
                    RMS_plot=True)
gru_model.reset_states()
# %%
# Convert the model to Tensorflow Lite and save.
with open(f'{model_loc}Model-№4-{profile}.tflite', 'wb') as f:
    f.write(
        tf.lite.TFLiteConverter.from_keras_model(
                model=gru_model
            ).convert()
        )