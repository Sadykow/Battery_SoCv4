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
import os

from parser.WindowGenerator import WindowGenerator                       # OS, SYS, argc functions
import pandas as pd             # File read
import matplotlib as mpl        # Plot functionality
import matplotlib.pyplot as plt
import tensorflow as tf         # Tensorflow and Numpy replacement
import tensorflow.experimental.numpy as tnp 
import logging

from sys import platform        # Get type of OS

from parser.DataGenerator import *
# %%
# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# Configurage logger and print basics
logging.basicConfig(level=logging.DEBUG,        
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
dataGenerator = DataGenerator(train_dir='Data/A123_Matt_Set',
                              valid_dir='Data/A123_Matt_Val',
                              test_dir='Data/A123_Matt_Test',
                              columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                                'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                                ],
                              PROFILE_range = 'FUDS')

# training = dataGenerator.train.loc[:, 
#                         ['Current(A)', 'Voltage(V)', 'Temperature (C)_1']]
# %%
window = WindowGenerator(Data=dataGenerator,
                        input_width=1, label_width=1, shift=0,
                        input_columns=['Current(A)', 'Voltage(V)', 'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,
                        includeTarget=False, normaliseLabal=False)
_, x_train, y_train = window.train
_, x_valid, y_valid = window.valid
# %%
def custom_loss(y_true, y_pred):
    #! No custom loss used in this implementation
    #!Used standard mean_squared_error()
    y_pred = tf.framework.ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = tf.framework.ops.math_ops.cast(y_true, y_pred.dtype)
    return tf.keras.backend.mean(tf.ops.math_ops.squared_difference(y_pred, y_true), axis=-1)


model_loc : str = 'Models/BinXiao2019/FUDS-models/'
iEpoch  : int = 0
p2 : int = 10
skipCompile1, skipCompile2 = False, False
try:
    for _, _, files in os.walk(model_loc):
        for file in files:
            if file.endswith('.ch'):
                iEpoch = int(os.path.splitext(file)[0])
    
    gru_model : tf.keras.models.Sequential = tf.keras.models.load_model(
            f'{model_loc}{iEpoch}',
            compile=False)
    print("Model Identefied. Continue training.")
except OSError as identifier:
    print("Model Not Found, creating new. {} \n".format(identifier))
    gru_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=x_train.shape[-2:],batch_size=1),
        tf.keras.layers.GRU(
            units=260, activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.0,
            recurrent_dropout=0.0, return_sequences=False, return_state=False,
            go_backwards=False, stateful=False, unroll=False, time_major=False,
            reset_after=True
        ),
        tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
        tf.keras.layers.Dense(units=12, activation='sigmoid'),
        tf.keras.layers.Dense(units=1,
                              kernel_initializer=tf.initializers.constant(0.5),
                              activation=None)
    ])

checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath =model_loc+'FUDS-checkpoints/checkpoint',
    monitor='val_loss', verbose=0,
    save_best_only=False, save_weights_only=False,
    mode='auto', save_freq='epoch', options=None,
)

mEpoch : int = 50
firtstEpoch : bool = True
while iEpoch < mEpoch:
    if (iEpoch<p2 and not skipCompile1):
        gru_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Nadam(learning_rate=10e-04,
                    beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Nadam'
                    ),
                metrics=[tf.metrics.MeanAbsoluteError(),
                         tf.metrics.RootMeanSquaredError()]
            )
        skipCompile1 = True
        print("\nOptimizer set: Nadam\n")
    elif (iEpoch>=mEpoch and not skipCompile2):
        gru_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adamax(learning_rate=10e-04,
                    beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Adamax'
                    ),
                metrics=[tf.metrics.MeanAbsoluteError(),
                         tf.metrics.RootMeanSquaredError()]
            )
        skipCompile2 = True
        print("\nOptimizer set: Adamax\n")
    iEpoch+=1
    print(f"Epoch {iEpoch}/{mEpoch}")
    
    history = gru_model.fit(x=x_train, y=y_train, epochs=1,
                        validation_data=(x_valid, y_valid),
                        callbacks=[checkpoints], batch_size=1
                        )#! Initially Batch size 1; 8 is safe to run - 137s
    gru_model.save(f'{model_loc}{iEpoch}')
    
    if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
        os.remove(f'{model_loc}{iEpoch-1}.ch')
    os.mknod(f'{model_loc}{iEpoch}.ch')
    
    # Saving history variable
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)
    # or save to csv:
    with open(f'{model_loc}history-FUDS.csv', mode='a') as f:
        if(firtstEpoch):
            hist_df.to_csv(f, index=False)
            firtstEpoch = False
        else:
            hist_df.to_csv(f, index=False, header=False)
    
    #! Run the Evaluate function
    if(iEpoch % 10 == 0):
        skip=1
        TAIL=y_valid.shape[0]
        PRED = gru_model.predict(x_valid)
        RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                    y_valid[::skip,-1]-PRED)))
        vl_test_time = range(0,PRED.shape[0])
        fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
        ax1.plot(vl_test_time[:TAIL:skip], y_valid[::skip,-1],
                label="True", color='#0000ff')
        ax1.plot(vl_test_time[:TAIL:skip],
                PRED,
                label="Recursive prediction", color='#ff0000')

        ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
        ax1.set_xlabel("Time Slice (s)", fontsize=16)
        ax1.set_ylabel("SoC (%)", fontsize=16)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(vl_test_time[:TAIL:skip],
                RMS,
                label="RMS error", color='#698856')
        ax2.fill_between(vl_test_time[:TAIL:skip],
                RMS[:,0],
                    color='#698856')
        ax2.set_ylabel('Error', fontsize=16, color='#698856')
        ax2.tick_params(axis='y', labelcolor='#698856')
        ax1.set_title("BinXiao LSTM Test 2019 - Valid dataset. FUDS-trained",
                    fontsize=18)
        ax1.legend(prop={'size': 16})
        ax1.set_ylim([-0.1,1.2])
        ax2.set_ylim([-0.1,1.6])
        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        val_perf = gru_model.evaluate(x=x_valid,
                                      y=y_valid,
                                      verbose=0)
        textstr = '\n'.join((
            r'$Loss =%.2f$' % (val_perf[0], ),
            r'$MAE =%.2f$' % (val_perf[1], ),
            r'$RMSE=%.2f$' % (val_perf[2], )))
        ax1.text(0.85, 0.75, textstr, transform=ax1.transAxes, fontsize=18,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.savefig(f'{model_loc}FUDS-val-{iEpoch}.svg')
# %%
# normaliseInput = False
# normaliseLabal = True
# includeTarget = False
# input_columns=['Current(A)', 'Voltage(V)', 'Temperature (C)_1']
# label_columns=['SoC(%)']
# batch=1
# float_dtype=tnp.float32

# @tf.autograph.experimental.do_not_convert
# def make_dataset(inputs : pd.DataFrame,
#                          labels : pd.DataFrame
#               ) -> tf.raw_ops.MapDataset:

#     tic : float = time.perf_counter()
#     if normaliseInput: # Normalise Inputs
#         data : pd.DataFrame = (inputs.copy(deep=True)-dataGenerator.get_Mean[0][input_columns])/dataGenerator.get_STD[0][input_columns]
#     else:
#         data : pd.DataFrame = (inputs.copy(deep=True))
    
#     if normaliseLabal: # Normalise Labels
#         data[label_columns] = (labels.copy(deep=True)-dataGenerator.get_Mean[1][label_columns])/dataGenerator.get_STD[1][label_columns]
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
#             batch_size=batch, shuffle=False,
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
#     print(f"\n\nData windowing took: {(time.perf_counter() - tic):.2f} seconds")
#     return ds, x, y

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
#     tf.print(inputs)
#     print(labels.shape)
#     # Slicing doesn't preserve static shape information, so set the shapes
#     # manually. This way the `tf.dataGenerator.Datasets` are easier to inspect.
#     inputs.set_shape([None, input_width, None])
#     inputs = tf.transpose(
#                     a=inputs,
#                     perm=[0,2,1],
#                     conjugate=False,name='SwapFeatureWithHistory')
#     labels.set_shape([None, label_width, None])
#     tf.print(inputs)
#     print(labels.shape)
#     return inputs, labels

# test_ds, test_x, test_y = make_dataset(inputs=dataGenerator.train[input_columns],
#             labels=dataGenerator.train_SoC[label_columns])
# %%