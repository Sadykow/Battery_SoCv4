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

# Compared with an LST<-based RNN model,a GRU-based RNN model has a simpler 
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
# %%
# Extract params
try:
    opts, args = getopt.getopt(sys.argv[1:],"hd:e:g:p:",
                    ["help", "debug=", "epochs=",
                     "gpu=", "profile="])
except getopt.error as err: 
    # output error, and return with an error code 
    print (str(err)) 
    print ('EXEPTION: Arguments requied!')
    sys.exit(2)

# opts = [('-d', 'False'), ('-e', '50'), ('-g', '1'), ('-p', 'FUDS')]
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
#! Select GPU for usage. CPU versions ignores it
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
window = WindowGenerator(Data=dataGenerator,
                        input_width=500, label_width=1, shift=0,
                        input_columns=['Current(A)', 'Voltage(V)', 'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,
                        includeTarget=False, normaliseLabal=False,
                        shuffleTraining=False)
_, xx_train, yy_train = window.train
_, xx_valid, yy_valid = window.valid

# Entire Training set
x_train = np.array(xx_train, copy=True, dtype=np.float32)
y_train = np.array(yy_train, copy=True, dtype=np.float32)

# For validation use same training
x_valid = np.array(xx_train[16800:25000,:,:], copy=True, dtype=np.float32)
y_valid = np.array(yy_train[16800:25000,:]  , copy=True, dtype=np.float32)

# For test dataset take the remaining profiles.
mid = int(xx_valid.shape[0]/2)+350
x_test_one = np.array(xx_valid[:mid,:,:], copy=True, dtype=np.float32)
y_test_one = np.array(yy_valid[:mid,:], copy=True, dtype=np.float32)
x_test_two = np.array(xx_valid[mid:,:,:], copy=True, dtype=np.float32)
y_test_two = np.array(yy_valid[mid:,:], copy=True, dtype=np.float32)
# %%
custom_loss = lambda y_true, y_pred: tf.keras.backend.mean(
            x=tf.math.squared_difference(
                    x=tf.cast(x=y_true, dtype=y_pred.dtype),
                    y=tf.convert_to_tensor(value=y_pred)
                ),
            axis=-1,
            keepdims=False
        )

file_name : str = os.path.basename(__file__)[:-3]
model_loc : str = f'Models/{file_name}/{profile}-models/'

iEpoch : int = 0
p2     : int = int(mEpoch/3)
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
        tf.keras.layers.InputLayer(input_shape=x_train.shape[-2:],
                                   batch_size=None),
        tf.keras.layers.GRU(    #?260 by BinXia, times by 2 or 3
            units=560, activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal', bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            recurrent_constraint=None, bias_constraint=None, dropout=0.2,
            recurrent_dropout=0.0, return_sequences=False, return_state=False,
            go_backwards=False, stateful=False, unroll=False, time_major=False,
            reset_after=True
        ),
        tf.keras.layers.Dense(units=1,
                              activation='sigmoid')
    ])
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
# %%
i_attempts : int = 0
n_attempts : int = 3
skip       : int = 1
firtstEpoch: bool = True
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
    
    history = gru_model.fit(x=x_train, y=y_train, epochs=1,
                        validation_data=(x_valid, y_valid),
                        callbacks=[nanTerminate],
                        batch_size=1, shuffle=True
                        )
    
    #? Dealing with NaN state. Give few trials to see if model improves
    if (tf.math.is_nan(history.history['loss'])):
        print('NaN model')
        while i_attempts < n_attempts:
            print(f'Attempt {i_attempts}')
            gru_model = tf.keras.models.clone_model(prev_model)
            #! Single compiler selection
            gru_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0005,
                        beta_1=0.9, beta_2=0.999, epsilon=10e-08, name='Adamax'
                        ),
                    metrics=[tf.metrics.MeanAbsoluteError(),
                            tf.metrics.RootMeanSquaredError(),
                            tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)]
                )
            history = gru_model.fit(x=x_train[:,:,:], y=y_train[:,:], epochs=1,
                            validation_data=None,
                            callbacks=[nanTerminate],
                            batch_size=1, shuffle=True
                            )
            if (not tf.math.is_nan(history.history['loss'])):
                print(f'Attempt {i_attempts} Passed')
                break
            i_attempts += 1
        if (i_attempts == n_attempts) \
                and (tf.math.is_nan(history.history['loss'])):
            print("Model reaced the optimim -- Breaking")
            break
        else:
            gru_model.save(filepath=f'{model_loc}{iEpoch}-{i_attempts}',
                            overwrite=True, include_optimizer=True,
                            save_format='h5', signatures=None, options=None,
                            save_traces=True
                )
            # gru_model.save_weights(f'{model_loc}weights/{iEpoch}-{i_attempts}')
            i_attempts = 0
            prev_model = tf.keras.models.clone_model(gru_model)
    else:
        gru_model.save(filepath=f'{model_loc}{iEpoch}',
                       overwrite=True, include_optimizer=True,
                       save_format='h5', signatures=None, options=None,
                       save_traces=True
                )
        # gru_model.save_weights(f'{model_loc}weights/{iEpoch}')
        prev_model = tf.keras.models.clone_model(gru_model)
    
    if os.path.exists(f'{model_loc}{iEpoch-1}.ch'):
        os.remove(f'{model_loc}{iEpoch-1}.ch')
    os.mknod(f'{model_loc}{iEpoch}.ch')
    
    # Saving history variable
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)
    # or save to csv:
    with open(f'{model_loc}history-{profile}.csv', mode='a') as f:
        if(firtstEpoch):
            hist_df.to_csv(f, index=False)
            firtstEpoch = False
        else:
            hist_df.to_csv(f, index=False, header=False)
    
    #! Run the Evaluate function
    PRED = gru_model.predict(x_valid,batch_size=1)
    RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(
                y_valid[::skip,]-PRED)))
    PERF = gru_model.evaluate(x=x_valid,
                              y=y_valid,
                              batch_size=1,
                              verbose=0)
    # otherwise the right y-label is slightly clipped
    predicting_plot(profile=profile, file_name='Model №2',
                    model_loc=model_loc,
                    model_type='GRU Train',
                    iEpoch=f'val-{iEpoch}',
                    Y=y_valid,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=PERF,
                    TAIL=y_valid.shape[0],
                    save_plot=True)
    if(PERF[-2] <=0.024): # Check thr RMSE
        print("RMS droped around 2.4%. Breaking the training")
        break

# %%
PRED = gru_model.predict(x_test_one, batch_size=1, verbose=1)
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(y_test_one[::,]-PRED)))
if profile == 'DST':
    predicting_plot(profile=profile, file_name='Model №2',
                    model_loc=model_loc,
                    model_type='GRU Test on US06', iEpoch=f'Test One-{iEpoch}',
                    Y=y_test_one,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=gru_model.evaluate(
                                    x=x_test_one,
                                    y=y_test_one,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_one.shape[0],
                    save_plot=True)
else:
    predicting_plot(profile=profile, file_name='Model №2',
                    model_loc=model_loc,
                    model_type='GRU Test on DST', iEpoch=f'Test One-{iEpoch}',
                    Y=y_test_one,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=gru_model.evaluate(
                                    x=x_test_one,
                                    y=y_test_one,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_one.shape[0],
                    save_plot=True)

PRED = gru_model.predict(x_test_two, batch_size=1, verbose=1)
RMS = (tf.keras.backend.sqrt(tf.keras.backend.square(y_test_two[::,]-PRED)))
if profile == 'FUDS':
    predicting_plot(profile=profile, file_name='Model №2',
                    model_loc=model_loc,
                    model_type='GRU Test on US06', iEpoch=f'Test Two-{iEpoch}',
                    Y=y_test_two,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=gru_model.evaluate(
                                    x=x_test_two,
                                    y=y_test_two,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_two.shape[0],
                    save_plot=True)
else:
    predicting_plot(profile=profile, file_name='Model №2',
                    model_loc=model_loc,
                    model_type='GRU Test on FUDS', iEpoch=f'Test Two-{iEpoch}',
                    Y=y_test_two,
                    PRED=PRED,
                    RMS=RMS,
                    val_perf=gru_model.evaluate(
                                    x=x_test_two,
                                    y=y_test_two,
                                    batch_size=1,
                                    verbose=1),
                    TAIL=y_test_two.shape[0],
                    save_plot=True)
# %%
# Convert the model to Tensorflow Lite and save.
with open(f'{model_loc}Model-№2-{profile}.tflite', 'wb') as f:
    f.write(
        tf.lite.TFLiteConverter.from_keras_model(
                model=gru_model
            ).convert()
        )
