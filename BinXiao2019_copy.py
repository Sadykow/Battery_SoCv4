#!/usr/bin/python
# %% [markdown]
# # # 2
# # # 
# # GRU for SoC by Bin Xiao - 2019
# 

# Data windowing has been used as per: Ψ ={X,Y} where:
#Here,X_k=[I(k),V(k),T(k)] and Y_k=[SoC(k)], whereI(k),V(k),T(k) and SoC(k) are
#the current, voltage, temperature and SoC ofthe battery as measured at time
#step k.

# Compared with an LSTM-based RNN model,a GRU-based RNN model has a simpler 
#structure and fewer parameters, thus making model training easier. The GRU 
#structure is shown in Fig. 1

#* The architecture consis of Inut layer, GRU hidden, fullt conected Dense
#*and Output. Dropout applied at hidden layer.
#* Dense fully connected uses sigmoind activation.

#* Loss function standard MSE, I think. By the looks of it.

#* 2 optimizers for that:
#*  Nadam (Nesterov momentum into the Adam) b1=0.99
#*Remark 1:The purpose of the pre-training phase is to endow the GRU_RNN model
#*with the appropriate parametersto capture the inherent features of the 
#*training samples. The Nadam algorithm uses adaptive learning rates and
#*approximates the gradient by means of the Nesterov momentum,there by ensuring
#*fast convergence of the pre-training process.
#*  AdaMax (Extension to adam)
#*Remark 2:The purpose of the fine-tuning phase is to further adjust the
#*parameters to achieve greater accuracy bymeans of the AdaMax algorithm, which
#*converges to a morestable value.
#* Combine those two methods: Ensemle ptimizer.

#* Data scaling was performed using Min-Max equation:
#* x′=(x−min_x)/(max_x−min_x) - values between 0 and 1.

#* Test were applied separetly at 0,30,50C
#*RMSE,MAX,MAPE,R2
# %%
import os                       # OS, SYS, argc functions
import pandas as pd             # File read
import matplotlib as mpl        # Plot functionality
import matplotlib.pyplot as plt
import tensorflow as tf         # Tensorflow and Numpy replacement
import tensorflow_addons as tfa
import numpy as np
import logging

from sys import platform        # Get type of OS

from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator

#! Temp Fix
import numpy as np
# %%
# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

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
    logging.info("GPU found and memory growth enabled") 
    
    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    logging.info("GPU found") 
    logging.debug(f"\nPhysical GPUs: {len(physical_devices)}"
                  f"\nLogical GPUs: {len(logical_devices)}")
#! For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float32')
# %%
profile : str = 'd_DST'
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

# training = dataGenerator.train.loc[:, 
#                         ['Current(A)', 'Voltage(V)', 'Temperature (C)_1']]

# %%
window = WindowGenerator(Data=dataGenerator,
                        input_width=1, label_width=1, shift=1,
                        input_columns=['Current(A)', 'Voltage(V)', 'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,
                        includeTarget=False, normaliseLabal=False,
                        shuffleTraining=False)
# ds_train, _, _ = window.train
# ds_valid, _, _ = window.valid
# _, ds_train, xx_train, yy_train = window.full_train
# _, ds_valid, xx_valid, yy_valid = window.full_valid
# x_train = np.array(xx_train)
# x_valid = np.array(xx_valid)
# y_train = np.array(yy_train)
# y_valid = np.array(yy_valid)
x_train, y_train = window.train
#x_valid, y_valid = window.valid

for i in range(0, len(y_train)):
    y_train[i] = (1-y_train[i])

# %%
def custom_loss(y_true, y_pred):
    #! No custom loss used in this implementation
    #!Used standard mean_squared_error()
    y_pred = tf.framework.ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = tf.framework.ops.math_ops.cast(y_true, y_pred.dtype)
    return tf.keras.backend.mean(tf.ops.math_ops.squared_difference(y_pred, y_true), axis=-1)

model_loc : str = f'Models/BinXiao2019/{profile}-models/'
iEpoch  : int = 0
p2 : int = 20
skipCompile1, skipCompile2 = False, False
try:
    for _, _, files in os.walk(model_loc):
        for file in files:
            if file.endswith('.ch'):
                iEpoch = int(os.path.splitext(file)[0])
    
    gru_model : tf.keras.models.Sequential = tf.keras.models.load_model(
            f'{model_loc}{iEpoch}',
            compile=True)
    print("Model Identefied. Continue training.")
except OSError as identifier:
    print("Model Not Found, creating new. {} \n".format(identifier))
    gru_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape=(8, 1, 3)),
        tf.keras.layers.GRU(    #?260 by BinXia, times by 2 or 3
            units=260, activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.2,
            recurrent_dropout=0.0, return_sequences=False, return_state=False,
            go_backwards=False, stateful=True, unroll=False, time_major=False,
            reset_after=True
        ),
        # tf.keras.layers.GRU(    #?260 by BinXia, times by 2 or 3
        #     units=260, activation='tanh', recurrent_activation='sigmoid',
        #     use_bias=True, kernel_initializer='glorot_uniform',
        #     recurrent_initializer='orthogonal', bias_initializer='zeros',
        #     kernel_regularizer=None,
        #     recurrent_regularizer=None, bias_regularizer=None,
        #     activity_regularizer=None, kernel_constraint=None,
        #     recurrent_constraint=None, bias_constraint=None, dropout=0.2,
        #     recurrent_dropout=0.0, return_sequences=False, return_state=False,
        #     go_backwards=False, stateful=True, unroll=False, time_major=False,
        #     reset_after=True
        # ),
        #tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
        #tf.keras.layers.Dense(units=12, activation='sigmoid'),
        # tf.keras.layers.TimeDistributed(
        #     tf.keras.layers.Dense(1,name="dense"))
        tf.keras.layers.Dense(units=1,
                              activation=None)
    ])

checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath =model_loc+f'{profile}-checkpoints/checkpoint',
    monitor='val_loss', verbose=0,
    save_best_only=False, save_weights_only=False,
    mode='auto', save_freq='epoch', options=None,
)

class ResetStatesCallback(tf.keras.callbacks.Callback):
    #? 230 by BinXiao, times by 2 or 3
    #? 126356 by 230
    max_len : int = 550
    counter : int = 0
    def __init__(self):
        self.counter = 0

    def on_batch_begin(self, batch, logs=None):
        if self.counter % self.max_len == 0:
            #print("\nReseting State\n")
            self.model.reset_states()
            self.counter = 0
        self.counter += 1
# %%
mEpoch : int = 100
firtstEpoch : bool = True
#!DEBUG
#iEpoch = 9
while iEpoch < mEpoch:
    if (iEpoch<=p2 and not skipCompile1):
        gru_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001,
                    beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Nadam'
                    ),
                metrics=[tf.metrics.MeanAbsoluteError(),
                         tf.metrics.RootMeanSquaredError(),
                         tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
            )
        skipCompile1 = True
        print("\nOptimizer set: Nadam\n")
    elif (iEpoch>p2 and not skipCompile2):
        gru_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0005,
                    beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Adamax'
                    ),
                metrics=[tf.metrics.MeanAbsoluteError(),
                         tf.metrics.RootMeanSquaredError(),
                         tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
            )
        skipCompile2 = True
        print("\nOptimizer set: Adamax\n")
    iEpoch+=1
    print(f"Epoch {iEpoch}/{mEpoch}")
    
    history = gru_model.fit(x=x_train[0][:-(x_train[0].shape[0]%230),:,:],
                            y=y_train[0][:-(x_train[0].shape[0]%230)], epochs=1,
                        #validation_data=(x_valid[0], y_valid[0]),
                        callbacks=[
                                checkpoints,
                                #ResetStatesCallback()
                            ],
                        batch_size=230, shuffle=True
                        )
    
    hist_df = pd.DataFrame(history.history)
    for i in range(1, len(x_train)):
        history = gru_model.fit(x=x_train[i][:-(x_train[i].shape[0]%230),:,:],
                                y=y_train[i][:-(x_train[i].shape[0]%230)], epochs=1,
                            #validation_data=(x_valid[0], y_valid[0]),
                            callbacks=[
                                    checkpoints,
                                    #ResetStatesCallback()
                                ],
                            batch_size=230, shuffle=True
                            )
        hist_df = hist_df.append(pd.DataFrame(history.history))
    gru_model.reset_states()
    # history = gru_model.fit(x=ds_train.take(362), epochs=1,
    #                     validation_data=ds_valid.take(73),
    #                     callbacks=[checkpoints], batch_size=1
    #                     )#! Initially Batch size 1; 8 is safe to run - 137s
    gru_model.save(f'{model_loc}{iEpoch}')
    gru_model.save_weights(f'{model_loc}weights/{iEpoch}')

    if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
        os.remove(f'{model_loc}{iEpoch-1}.ch')
    os.mknod(f'{model_loc}{iEpoch}.ch')
    
    # Saving history variable
    # convert the history.history dict to a pandas DataFrame:     
    #hist_df = pd.DataFrame(history.history)
    # or save to csv:
    with open(f'{model_loc}history-{profile}.csv', mode='a') as f:
        if(firtstEpoch):            
            pd.DataFrame(data=hist_df.mean()).T.to_csv(f, index=False)
            firtstEpoch = False
        else:
            pd.DataFrame(data=hist_df.mean()).T.to_csv(f, index=False, header=False)
    
    
    # #! Run the Evaluate function
    # if(iEpoch % 10 == 0):
        # skip=1
        # TAIL=y_valid[1].shape[0]
        # gru_model.reset_states()
        # PRED = gru_model.predict(x_valid[1], batch_size=1)
        # RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
        #             y_valid[1][::skip,]-PRED)))
        # vl_test_time = range(0,PRED.shape[0])
        # fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
        # ax1.plot(vl_test_time[:TAIL:skip], y_valid[1][::skip],
        #         label="True", color='#0000ff')
        # ax1.plot(vl_test_time[:TAIL:skip],
        #         PRED,
        #         label="Recursive prediction", color='#ff0000')

        # ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
        # ax1.set_xlabel("Time Slice (s)", fontsize=16)
        # ax1.set_ylabel("SoC (%)", fontsize=16)

        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        # ax2.plot(vl_test_time[:TAIL:skip],
        #         RMS[:,0],
        #         label="RMS error", color='#698856')
        # ax2.fill_between(vl_test_time[:TAIL:skip],
        #         RMS[:,0],
        #             color='#698856')
        # ax2.set_ylabel('Error', fontsize=16, color='#698856')
        # ax2.tick_params(axis='y', labelcolor='#698856')
        # ax1.set_title("BinXiao GRU Test 2019 - Valid dataset. {profile}-trained",
        #             fontsize=18)
        # ax1.legend(prop={'size': 16})
        # ax1.set_ylim([-0.1,1.2])
        # ax2.set_ylim([-0.1,1.6])
        # fig.tight_layout()  # otherwise the right y-label is slightly clipped

        # gru_model.reset_states()
        # val_perf = gru_model.evaluate(x=x_valid[1],
        #                               y=y_valid[1],
        #                               batch_size=1,
        #                               verbose=0)
        # textstr = '\n'.join((
        #     r'$Loss =%.2f$' % (val_perf[0], ),
        #     r'$MAE =%.2f$' % (val_perf[1], ),
        #     r'$RMSE=%.2f$' % (val_perf[2], )))
        # ax1.text(0.85, 0.75, textstr, transform=ax1.transAxes, fontsize=18,
        #         verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    #     fig.savefig(f'{model_loc}{profile}-val-{iEpoch}.svg')
# %%
# skip=1
# start = 0
# x_test = x_train[2][:,:,:]
# y_test = y_train[2]
# TAIL=y_test.shape[0]
# gru_model.reset_states()
# PRED = gru_model.predict(x_test, batch_size=1)
# RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
#             y_test[::skip,]-PRED)))
# vl_test_time = range(0,PRED.shape[0])

# fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
# ax1.plot(vl_test_time[:TAIL:skip], y_test[::skip],
#         label="True", color='#0000ff')
# ax1.plot(vl_test_time[:TAIL:skip],
#         PRED,
#         label="Recursive prediction", color='#ff0000')

# ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
# ax1.set_xlabel("Time Slice (s)", fontsize=16)
# ax1.set_ylabel("SoC (%)", fontsize=16)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax2.plot(vl_test_time[:TAIL:skip],
#         RMS,
#         label="RMS error", color='#698856')
# # ax2.fill_between(vl_test_time[:TAIL:skip],
# #         RMS[:,0],
# #             color='#698856')
# ax2.set_ylabel('Error', fontsize=16, color='#698856')
# ax2.tick_params(axis='y', labelcolor='#698856')
# ax1.set_title(f"BinXiao GRU Test 2017 - Train dataset. {profile}-trained",
#             fontsize=18)
# ax1.legend(prop={'size': 16})
# ax1.set_ylim([-0.1,1.2])
# ax2.set_ylim([-0.1,1.6])
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# gru_model.reset_states()
# val_perf = gru_model.evaluate(x=x_test,
#                                 y=y_test,
#                                 batch_size=1,
#                                 verbose=0)
# textstr = '\n'.join((
#     r'$Loss =%.2f$' % (val_perf[0], ),
#     r'$MAE =%.2f$' % (val_perf[1], ),
#     r'$RMSE=%.2f$' % (val_perf[2], )))
# ax1.text(0.85, 0.75, textstr, transform=ax1.transAxes, fontsize=18,
#         verticalalignment='top',
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
# fig.savefig(f'{model_loc}{profile}-train-{iEpoch}.svg')



# normaliseInput = False
# normaliseLabal = True
# includeTarget = False
# input_columns=['Current(A)', 'Voltage(V)', 'Temperature (C)_1']
# label_columns=['SoC(%)']
# batch=5
# float_dtype=tnp.float32

# @tf.autograph.experimental.do_not_convert
# def make_dataset(inputs : pd.DataFrame,
#                          labels : pd.DataFrame
#               ) -> tf.raw_ops.MapDataset:

#     tic : float = time.perf_counter()
#     if normaliseInput: # Normalise Inputs
#         data : pd.DataFrame = (inputs.copy(deep=True))
#     else:
#         data : pd.DataFrame = (inputs.copy(deep=True))
    
#     if normaliseLabal: # Normalise Labels
#         data[label_columns] = (labels.copy(deep=True))
#     else:
#         data[label_columns] = (labels.copy(deep=True))

#     data = data[input_columns + label_columns] # Ensure order
#     data = tnp.array(val=data.values,
#             dtype=float_dtype, copy=True, ndmin=0)

#     ds : tf.raw_ops.BatchDataset = \
#           tf.keras.preprocessing.timeseries_dataset_from_array(
#             data=data, targets=None,
#             sequence_length=total_window_size, sequence_stride=1,
#             sampling_rate=1,
#             batch_size=1, shuffle=False,
#             seed=None, start_index=None, end_index=None
#         )

#     ds : tf.raw_ops.MapDataset = ds.map(split_window)
#     x : tnp.ndarray = tnp.asarray(list(ds.map(
#                                 lambda x, _: x[0,:,:]
#                               ).as_numpy_iterator()
#                           ))
#     y : tnp.ndarray = tnp.asarray(list(ds.map(
#                                 lambda _, y: y[0]
#                               ).as_numpy_iterator()
#                           ))
#     #! Batching HERE manually
#     batched_x : tnp.ndarray = tnp.reshape(x[0:0+batch,:,:], (1,batch,3))
#     batched_y : tnp.ndarray = tnp.reshape(y[0:0+batch,:,:], (1,batch))
#     #print(batched_x)
#     for i in range(1, x.shape[0]-batch+1):
#         batched_x = tnp.append(arr=batched_x,
#                                values=tnp.reshape(x[i:i+batch,:,:], (1,batch,3)),
#                                axis=0)
#         batched_y = tnp.append(arr=batched_y,
#                                values=tnp.reshape(y[i:i+batch,:,:], (1,batch)),
#                                axis=0)
#         #print(i)
#         #print("Manual Batching Complete")

#     print(f"\n\nData windowing took: {(time.perf_counter() - tic):.2f} seconds")
#     return ds, batched_x, batched_y

# input_width=1
# label_width=1
# shift=0
# total_window_size = input_width+shift
# input_slice = slice(0, input_width)
# labels_slice = slice(total_window_size-label_width, None)

# @tf.autograph.experimental.do_not_convert
# def split_window(features : tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
#     if includeTarget:
#       inputs : tf.Tensor=features[:,input_slice,:]
#     else:
#       #! Fix exlude similarly to tf.stack()
#       inputs : tf.Tensor=features[:,input_slice,:-len(label_columns)]
    
#     labels : tf.Tensor = features[:, labels_slice, :]
#     #labels = tf.stack([labels[:, :, -2], labels[:, :, -1]], axis=-1)
#     #labels = tf.stack([labels[:, :, -1]], axis=-1)
#     labels = tf.stack(
#                 [labels[:, :, -i]
#                     for i in range(len(label_columns),0,-1)], axis=-1)
#     #tf.print(inputs)
#     #print(labels.shape)
#     # Slicing doesn't preserve static shape information, so set the shapes
#     # manually. This way the `tf.dataGenerator.Datasets` are easier to inspect.
#     inputs.set_shape([None, input_width, None])
#     inputs = tf.transpose(
#                     a=inputs,
#                     perm=[0,2,1],
#                     conjugate=False,name='SwapFeatureWithHistory')
#     labels.set_shape([None, label_width, None])
#     #tf.print("\nInputs: ", inputs)
#     #tf.print("Labels: ", labels, "\n")
#     #print(labels.shape)
#     return inputs, labels

# df = pd.DataFrame({'Current(A)': np.arange(0,100,1),
#                    'Voltage(V)': np.arange(0,1000,10),
#                    'Temperature (C)_1': np.arange(0,10000,100)})
# df_label = pd.DataFrame({'SoC(%)': np.arange(1,101,1)})
# test_ds, test_x, test_y = make_dataset(inputs=df,
#                                         labels=df_label)
# %%
#! Experement
# def custom_loss_two(y_true, y_pred):
#     #print(f"True: {y_true[0]}" ) # \nvs Pred: {tf.make_ndarray(y_pred)}")
#     print(f"\nY_TRUE: {y_true}\n")
#     print(f"Y_PRED: {y_pred}\n")
#     loss = tf.keras.backend.square(y_true - y_pred)/2
#     #print("\nInitial Loss: {}".format(loss))    
#     loss = tf.keras.backend.sum(loss, axis=1)
#     print(f"LOSS: {loss}\n")
#     #print(  "Summed  Loss: {}\n".format(loss))
#     return loss

# gru_model.compile(loss=custom_loss_two,
#         optimizer=tf.keras.optimizers.Adam(learning_rate=10e-04,
#             beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Adam'
#             ),
#         metrics=[tf.metrics.MeanAbsoluteError(),
#                     tf.metrics.RootMeanSquaredError()]
#     )
# batch=500
# offset = x_train.shape[0]%batch
# mean_tr_loss = np.empty(shape=(x_train.shape[0]-offset), dtype=np.float32, order='C')
# mean_tr_mse = np.empty(shape=(x_train.shape[0]-offset), dtype=np.float32, order='C')
# mean_tr_rmse = np.empty(shape=(x_train.shape[0]-offset), dtype=np.float32, order='C')
# #for i in range(1, x_train.shape[0]):
# #print("Time: {}".format((x_train.shape[0]-offset)/2.5/60/60))
# #for i in tf.range(0, x_train.shape[0]-offset):
# for i in tf.range(0, 1000):
#     #tic : float = time.perf_counter()
#     for j in tf.range(0, batch):
#         tr_loss, tr_mse, tr_rmse = gru_model.train_on_batch(
#                             x=x_train[(i*batch)+j:(i*batch)+j+1,:,:],
#                             y=y_train[(i*batch)+j,:,:],
#                             sample_weight=None, class_weight=None,
#                             reset_metrics=True, return_dict=False
#                         )
#         mean_tr_loss[(i*batch)+j] = tr_loss
#         mean_tr_mse[(i*batch)+j] = tr_mse
#         mean_tr_rmse[(i*batch)+j] = tr_rmse
#     gru_model.reset_states()
#     #print(f"Sample {i}: {time.perf_counter()-tic}")
# print('Loss training = {}'.format(np.mean(mean_tr_loss)))
# print('MSE training = {}'.format(np.mean(mean_tr_mse)))
# print('RMSE training = {}'.format(np.mean(mean_tr_rmse)))
# print('___________________________________')
# # %%
# gru_model.compile(loss=tf.keras.losses.MeanSquaredError(),
#         optimizer=tf.keras.optimizers.Adam(learning_rate=10e-04,
#             beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Adam'
#             ),
#         metrics=[tf.metrics.MeanAbsoluteError(),
#                     tf.metrics.RootMeanSquaredError()]
#     )
# mean_tr_loss = np.empty(shape=(x_train.shape[0]), dtype=np.float32, order='C')
# mean_tr_mse = np.empty(shape=(x_train.shape[0]), dtype=np.float32, order='C')
# mean_tr_rmse = np.empty(shape=(x_train.shape[0]), dtype=np.float32, order='C')
# tic : float = time.perf_counter()
# for i in range(1, int(x_train.shape[0]/2)):
#     tr_loss, tr_mse, tr_rmse = gru_model.train_on_batch(
#                         x=x_train[i-1:i,:,:],
#                         #x=x_train[i,:,:],
#                         y=y_train[i-1:i,:,0],
#                         sample_weight=None, class_weight=None,
#                         reset_metrics=True, return_dict=False
#                     )
#     mean_tr_loss[(i-1)] = tr_loss
#     mean_tr_mse[(i-1)] = tr_mse
#     mean_tr_rmse[(i-1)] = tr_rmse
#     gru_model.reset_states()
# print(f"Single Batch: {time.perf_counter()-tic}")
# print('Loss training = {}'.format(np.mean(mean_tr_loss)))
# print('MSE training = {}'.format(np.mean(mean_tr_mse)))
# print('RMSE training = {}'.format(np.mean(mean_tr_rmse)))
# print('___________________________________')
# # %%
# mean_te_loss = []
# mean_te_mse = []
# mean_te_rmse = []
# for i in range(1, len(x_valid.shape[0])):
#     for j in range(1, len(x_valid.shape[1])):
#         te_loss, te_mse, te_rmse = gru_model.test_on_batch(
#                             x=x_valid[i-1:i,j-1:j,:],
#                             y=y_valid[i-1:i,j-1:j,:],
#                             sample_weight=None, reset_metrics=True,
#                             return_dict=False
#                         )
#         mean_te_loss.append(te_loss)
#         mean_te_mse.append(te_mse)
#         mean_te_rmse.append(te_rmse)
#     gru_model.reset_states()

# # for j in range(max_len):
# #     y_pred = model.predict_on_batch(np.expand_dims(np.expand_dims(X_test[i][j], axis=1), axis=1))
# # model.reset_states()

# print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
# print('loss testing = {}'.format(np.mean(mean_te_loss)))
# print('___________________________________')

# epochs = 30
# for e in range(epochs):
#     print(f"Epoch {e}/{epochs}")
#     for (x, y) in zip(x_train[:2], y_train[:2]):
#         test_model.fit(x, y, batch_size=1,shuffle=False)
#         test_model.reset_states()
# for (x, y) in zip(x_train[:2], y_train[:2]):
#     plt.plot(test_model.predict(x, batch_size=1)+0.55)
#     plt.plot(y)
#     test_model.reset_states()

# %%
test_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape=(1, 1, 3)),
        tf.keras.layers.GRU(    #?260 by BinXia, times by 2 or 3
            units=260, activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.2,
            recurrent_dropout=0.0, return_sequences=False, return_state=False,
            go_backwards=False, stateful=True, unroll=False, time_major=False,
            reset_after=True
        ),
        tf.keras.layers.Dense(units=1,
                              activation=None)
    ])
test_model.compile(loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001,
        beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Nadam'
        ),
    metrics=[tf.metrics.MeanAbsoluteError(),
                tf.metrics.RootMeanSquaredError(),
                tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
)
epochs : int = 20
for e in range(epochs):
    print(f'Epochs {e+1}/{epochs}')
    for (x, y) in zip(x_train[:], y_train[:]):
        test_model.fit(x[:,:,:],
                       y[:],
                       batch_size=1, shuffle=False)
        test_model.reset_states()
for (x, y) in zip(x_train[:], y_train[:]):
    plt.plot(test_model.predict(x[:], batch_size=1))
    plt.plot(y)
    test_model.reset_states()
# %%
test_model.compile(loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0005,
            beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Adamax'
            ),
        metrics=[tf.metrics.MeanAbsoluteError(),
                    tf.metrics.RootMeanSquaredError(),
                    tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
    )
epochs : int = 80
for e in range(epochs):
    print(f'Epochs {e+1}/{epochs}')
    for (x, y) in zip(x_train[:], y_train[:]):
        test_model.fit(x[:,:,:],
                       y[:],
                       batch_size=1, shuffle=True, verbose=0)
        test_model.reset_states()
for (x, y) in zip(x_train[:], y_train[:]):
    plt.plot(test_model.predict(x[:], batch_size=1))
    plt.plot(y)
    test_model.reset_states()