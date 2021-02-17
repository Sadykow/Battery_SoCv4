# %%
import numpy as np
#from standard_plots import *
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

#import theano
# %% [markdown]
# # LSTMs
# Origin location:
# https://github.com/tykimos/tykimos.github.io/blob/a40bcc65952025bfb23c17060333231798739e26/_writing/cosine_LSTM.ipynb
# %%
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
# Create a dataset which will be experimented on
dataset : np.ndarray = np.cos(np.arange(1000)*(20*np.pi/1000))[:,None]
def_type = np.float32
plt.plot(dataset)
plt.title("Raw Dataset")
# %% [markdown]
# ## Window of 20 time steps
# %%
def create_dataset(dataset : np.ndarray, look_back : int = 1
                    ) -> tuple[np.ndarray, np.ndarray]:
    """ Convert an array if values into a dataset matrix

    Args:
        dataset (np.ndarray): Input array
        look_back (int, optional): History size. Defaults to 1.

    Returns:
        tuple[np.ndarray, np.ndarray]: Returns X and Y sets
    """
    d_len : int = len(dataset)-look_back
    dataX : np.ndarray = np.zeros(shape=(d_len, look_back, 1),
                                  dtype=def_type)
    dataY : np.ndarray = np.zeros(shape=(d_len),
                                  dtype=def_type)
    for i in range(0, d_len):
        dataX[i,:,:] = dataset[i:(i+look_back), :]
        dataY[i]     = dataset[i + look_back  , 0]
    return dataX, dataY

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

look_back : int = 20
scaler : MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
plt.plot(dataset)
plt.title("Scaled with MinMaxScaler")

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

trX, trY = create_dataset(train, look_back)
tsX, tsY = create_dataset(test, look_back)

# trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


# %%
print("Data Shapes:\n"
      "Train {}\n"
      "TrainX {}\n"
      "TrainY {}\n".format(train.shape,trX.shape, trY.shape)
      )

# %%
batch_size = 1
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(look_back, 1)),
    tf.keras.layers.LSTM(units=32, input_dim=1),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None), #! Specefying additional params breaks prediction
    tf.keras.layers.Dense(units=1,
        #! Setting activation function to None breaks during prediction
                        activation=None
                        )
])
#! Below is better with hell!!
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.LSTM(32,input_dim=1))
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dense(1))
print(model.summary())

model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=[tf.metrics.MeanAbsoluteError(),
                       tf.metrics.RootMeanSquaredError()]
            )
model.fit(trX, trY, epochs=100, batch_size=batch_size, verbose=1,
            callbacks = [
                # tf.keras.callbacks.EarlyStopping(monitor='loss',
                #                                 min_delta=0, 
                #                                 patience=200,
                #                                 verbose=1,
                #                                 mode='min'),

                tf.keras.callbacks.ModelCheckpoint('Models/cosine_lstm', 
                                            monitor='root_mean_squared_error',
                                            save_best_only=True, 
                                            mode='min',
                                            verbose=1)
                    ]
    )
best_model = tf.keras.models.load_model('Models/cosine_lstm', compile=False)
best_model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=[tf.metrics.MeanAbsoluteError(),
                       tf.metrics.RootMeanSquaredError()])
# %%
print("Latest model:")
model.evaluate(trX, trY, batch_size=batch_size, verbose=1)
model.evaluate(tsX[:252], tsY[:252], 
                            batch_size=batch_size, verbose=1)
print("\n\nBest model:")
best_model.evaluate(trX, trY, batch_size=batch_size, verbose=1)
best_model.evaluate(tsX[:252], tsY[:252], 
                            batch_size=batch_size, verbose=1)

# %%
# His
look_ahead = 250
trainPredict = [np.vstack([trX[-1][1:], trY[-1]])]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = model.predict(np.array([trainPredict[-1]]), batch_size=batch_size)
    predictions[i] = prediction
    trainPredict.append(np.vstack([trainPredict[-1][1:],prediction]))
plt.figure(figsize=(12,5))
# plt.plot(np.arange(len(trainX)),np.squeeze(trainX))
# plt.plot(np.arange(200),scaler.inverse_transform(np.squeeze(trainPredict)[:,None][1:]))
# plt.plot(np.arange(200),scaler.inverse_transform(np.squeeze(testY)[:,None][:200]),'r')
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),dataset[train_size:(train_size+look_ahead)],
        label="test function")
plt.legend()
plt.show()
# %%
# Mine
look_ahead = 250
# Stack the last sample together with X and Y.
trainPredict = np.array([np.vstack([trX[-1][1:],
                           trY[-1]])])   # Size 20
best_predictions = np.zeros((look_ahead,1))
for i in range(0, look_ahead):
    best_prediction = best_model.predict(trainPredict[-1:,:,:], batch_size=batch_size)
    best_predictions[i] = best_prediction
    #trainPredict = (np.append([trainPredict[-1:,1:,:],np.expand_dims(prediction,axis=1)]))
    trainPredict = np.reshape(np.append(trainPredict[-1,1:], best_prediction),
                              (1,20,1))
plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),best_predictions,'r',label="best_prediction")
plt.plot(np.arange(look_ahead),dataset[train_size:(train_size+look_ahead)],
        label="test function")
plt.legend()
plt.show()

# %% [markdown]
# ## Stateful LSTMs

# %%
# look_back = 20
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

# # split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)

# trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# %%
# Calculate neurons smartly

# %%
stateful_model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(batch_input_shape=(batch_size, look_back, 1)),
    tf.keras.layers.LSTM(units=32, stateful=True, return_sequences=False),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=1,
                            activation=None)
])
print(stateful_model.summary())

stateful_model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=[tf.metrics.MeanAbsoluteError(),
                       tf.metrics.RootMeanSquaredError()]
            )
for i in range(100):    
    stateful_model.fit(trX, trY, epochs=1, batch_size=batch_size, verbose=1,
            shuffle=False
    )
    stateful_model.reset_states()
# %%
print("Latest model:")
stateful_model.evaluate(trX, trY, batch_size=batch_size, verbose=1)
stateful_model.evaluate(tsX[:252], tsY[:252], 
                            batch_size=batch_size, verbose=1)
# %%
look_ahead = 250
trainPredict = [np.vstack([trX[-1][1:], trY[-1]])]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = stateful_model.predict(np.array([trainPredict[-1]]),
                                batch_size=batch_size)
    predictions[i] = prediction
    trainPredict.append(np.vstack([trainPredict[-1][1:],prediction]))

plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),dataset[train_size:(train_size+look_ahead)],
        label="test function")
plt.legend()
plt.show()

# %% [markdown]
# ## Stateful LSTMs with wider window

# %%
look_back = 40
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

# # split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

trX, trY = create_dataset(train, look_back)
tsX, tsY = create_dataset(test, look_back)

# trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


# %%
#get_ipython().run_cell_magic('time', '', 'theano.config.compute_test_value = "ignore"\n# create and fit the LSTM network\nbatch_size = 1\nmodel = Sequential()\n# model.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))\n# model.add(Dropout(0.3))\n# model.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))\n# model.add(Dropout(0.3))\nmodel.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True))\nmodel.add(Dropout(0.3))\nmodel.add(Dense(1))\nmodel.compile(loss=\'mean_squared_error\', optimizer=\'adam\')\nfor i in range(200):\n    model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=0, shuffle=False)\n    model.reset_states()')
batch_size = 1
stateful_model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(batch_input_shape=(batch_size, look_back, 1)),
    # tf.keras.layers.LSTM(32, batch_input_shape=(batch_size, look_back, 1),
    #                      stateful=True, return_sequences=True),
    # tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    # tf.keras.layers.LSTM(32, batch_input_shape=(batch_size, look_back, 1),
    #                     stateful=True, return_sequences=True),
    # tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(units=32, stateful=True, return_sequences=False),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=1,
                            activation=None)
])
print(stateful_model.summary())
stateful_model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=[tf.metrics.MeanAbsoluteError(),
                       tf.metrics.RootMeanSquaredError()]
            )
for i in range(100):    
    stateful_model.fit(trX, trY, epochs=1, batch_size=batch_size, verbose=1,
            shuffle=False
    )
    stateful_model.reset_states()
# %%
stateful_model.evaluate(trX, trY, batch_size=batch_size, verbose=1)
stateful_model.evaluate(tsX[:252], tsY[:252], 
                            batch_size=batch_size, verbose=1)
# %%
look_ahead = 250
trainPredict = [np.vstack([trX[-1][1:], trY[-1]])]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = stateful_model.predict(np.array([trainPredict[-1]]),
                                batch_size=batch_size)
    predictions[i] = prediction
    trainPredict.append(np.vstack([trainPredict[-1][1:],prediction]))

plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),dataset[train_size:(train_size+look_ahead)],
        label="test function")
plt.legend()
plt.show()
# %% [markdown]
# ## Stateful LSTMs, Stacked

# %%
batch_size = 1
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(batch_input_shape=(batch_size, look_back, 1)),
    tf.keras.layers.LSTM(units=32, stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(units=32, stateful=True, return_sequences=False),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=1, activation=None)
])
print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=[tf.metrics.MeanAbsoluteError(),
                       tf.metrics.RootMeanSquaredError()]
            )
for i in range(200):
    model.fit(trX, trY, epochs=1, batch_size=batch_size,
                verbose=1, shuffle=False)
    model.reset_states()
# %%
model.evaluate(trX, trY, batch_size=batch_size, verbose=1)
model.evaluate(tsX[:252], tsY[:252], 
                            batch_size=batch_size, verbose=1)
# %%
look_ahead = 250
trainPredict = [np.vstack([trX[-1][1:], trY[-1]])]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = model.predict(np.array([trainPredict[-1]]),
                                batch_size=batch_size)
    predictions[i] = prediction
    trainPredict.append(np.vstack([trainPredict[-1][1:],prediction]))
plt.figure(figsize=(12,5))
# plt.plot(np.arange(len(trainX)),np.squeeze(trainX))
# plt.plot(np.arange(200),scaler.inverse_transform(np.squeeze(trainPredict)[:,None][1:]))
# plt.plot(np.arange(200),scaler.inverse_transform(np.squeeze(testY)[:,None][:200]),'r')
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),dataset[train_size:(train_size+look_ahead)],
            label="test function")
plt.legend()
plt.show()

# %% [markdown]
# ## Stateful LSTM stacked DEEPER!

# %%
dataset = np.cos(np.arange(1000)*(20*np.pi/1000))[:,None]
look_back : int = 1
scaler : MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

trX, trY = create_dataset(train, look_back)
tsX, tsY = create_dataset(test, look_back)

#N_s: int = len(trX) # number of samples in the training data.
#alpha: int = 2 # a - (2-10)
# N_i: int = trX.shape[-2:][0] \
#             * trX.shape[-2:][1] # Ni - Number of input neurons
#N_o: int = 1 # No - Number of output neorons.
    
# print(f"№ of Samples in the training data: {N_s},\n"
#     f"\t\t\t\t    Chosen Scaling factor: {alpha}, \n"
#     f"\t\t\t\t  № of Input neurons: {N_i}, \n"
#     f"\t\t\t\t  № of Output neurons: {N_o}.")
#hidden_nodes =  roundup(N_s / (alpha * (N_i+N_o)))
#del N_s, alpha, N_i, N_o
#! Alpha 6 gives best so far
h_nodes : int = roundup(len(trX) / (6 * ((look_back * 1)+1)))
print(f"The number of hidden nodes is {h_nodes}.")

batch_size = 1
triple_model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(batch_input_shape=(batch_size, look_back, 1)),
    tf.keras.layers.LSTM(units=h_nodes, stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(units=h_nodes, stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(units=int(h_nodes/2), stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(units=int(h_nodes/2), stateful=True, return_sequences=False),
    tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=1, activation=None)
])
# look_back : int = 40
# triple_model = tf.keras.models.Sequential([
#     tf.keras.layers.InputLayer(batch_input_shape=(batch_size, look_back, 1)),
#     tf.keras.layers.LSTM(units=32, stateful=True, return_sequences=True),
#     tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
#     tf.keras.layers.LSTM(units=32, stateful=True, return_sequences=True),
#     tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
#     tf.keras.layers.LSTM(units=32, stateful=True, return_sequences=False),
#     tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
#     tf.keras.layers.Dense(units=1, activation=None)
# ])
print(triple_model.summary())
triple_model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=[tf.metrics.MeanAbsoluteError(),
                       tf.metrics.RootMeanSquaredError()]
            )
epochs : int = 250
for i in range(1,epochs+1):
    #print(f'Epoch {i}/{epochs}')
    triple_model.fit(trX, trY, epochs=1, batch_size=batch_size,
                verbose=1, shuffle=False)
    triple_model.reset_states()

triple_model.evaluate(tsX[:252], tsY[:252], 
                            batch_size=batch_size, verbose=1)
triple_model.reset_states()

triple_model.evaluate(trX, trY, batch_size=batch_size, verbose=1)
look_ahead : int = 250
trainPredict = [np.vstack([trX[-1][1:], trY[-1]])]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = triple_model.predict(np.array([trainPredict[-1]]),
                                batch_size=batch_size)
    predictions[i] = prediction
    trainPredict.append(np.vstack([trainPredict[-1][1:],prediction]))

plt.figure(figsize=(12,5))
# plt.plot(np.arange(len(trainX)),np.squeeze(trainX))
# plt.plot(np.arange(200),scaler.inverse_transform(np.squeeze(trainPredict)[:,None][1:]))
# plt.plot(np.arange(200),scaler.inverse_transform(np.squeeze(testY)[:,None][:200]),'r')
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),dataset[train_size:(train_size+look_ahead)],
            label="test function")
plt.legend()
plt.show()
#triple_model.save('Models/cosine_lstm')

# %% [markdown]
# ## Normal Deep Learning in Keras

# %%
#get_ipython().run_cell_magic('time', '', 'trainX = np.squeeze(trainX)\ntestX = np.squeeze(testX)\ntheano.config.compute_test_value = "ignore"\n# create and fit the LSTM network\nbatch_size = 1\nmodel = Sequential()\nmodel.add(Dense(output_dim=32,input_dim=40,activation="relu"))\nmodel.add(Dropout(0.3))\nfor i in range(2):\n    model.add(Dense(output_dim=32,activation="relu"))\n    model.add(Dropout(0.3))\nmodel.add(Dense(1))\nmodel.compile(loss=\'mean_squared_error\', optimizer=\'adagrad\')\nmodel.fit(trainX, trainY, nb_epoch=100, batch_size=32, verbose=0)')
trainX = np.squeeze(trX)
testX = np.squeeze(tsX)
batch_size = 1
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=32,input_dim=40,activation="relu"),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=32,activation="relu"),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=1,
                            activation=None)
])
# model = Sequential()
# model.add(Dense(output_dim=32,input_dim=40,activation="relu"))
# model.add(Dropout(0.3))
# for i in range(2):
#     model.add(Dense(output_dim=32,activation="relu"))
#     model.add(Dropout(0.3))
# model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adagrad')
model.fit(trX, trY, epochs=100, batch_size=32, verbose=0)

trainScore = model.evaluate(trainX, trY, batch_size=batch_size, verbose=0)
print('Train Score: ', trainScore)
testScore = model.evaluate(testX[:252], tsY[:252], batch_size=batch_size,
                            verbose=0)
print('Test Score: ', testScore)

look_ahead = 250
xval = np.hstack([trainX[-1][1:], trY[-1]])[None,:]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = model.predict(xval, batch_size=32)
    predictions[i] = prediction
    xval = np.hstack([xval[:,1:],prediction])

plt.figure(figsize=(12,5))
# plt.plot(np.arange(len(trainX)),np.squeeze(trainX))
# plt.plot(np.arange(200),scaler.inverse_transform(np.squeeze(trainPredict)[:,None][1:]))
# plt.plot(np.arange(200),scaler.inverse_transform(np.squeeze(testY)[:,None][:200]),'r')
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),dataset[train_size:(train_size+look_ahead)],
            label="test function")
plt.legend()
plt.show()
# %%
import pandas as pd
# Create a dataset which will be experimented on
dataset : np.ndarray = np.cos(np.arange(1200)*(20*np.pi/1000))[:,None]
def_type = np.float32
plt.plot(dataset[:100])
plt.title("Raw Dataset")
look_back : int = 1
scaler : MinMaxScaler = MinMaxScaler(feature_range=(0, 1))

b_dataframe : pd.DataFrame = pd.DataFrame(
            data=dataset[0:100,0],
            columns=['batch_0'],dtype=np.float32
            )
for i in range(1,12):
    b_dataframe[f'batch_{i}'] = dataset[i*100:i*100+100]

def create_Batch_dataset(dataset : np.ndarray, look_back : int = 1
                    ) -> tuple[np.ndarray, np.ndarray]:
    
    d_len : int = dataset.shape[0]-look_back
    batch : int = dataset.shape[1]

    dataX : np.ndarray = np.zeros(shape=(d_len, batch, look_back, 1),
                                  dtype=np.float32)
    dataY : np.ndarray = np.zeros(shape=(d_len, batch),
                                  dtype=np.float32)
    for i in range(0, d_len):
        for j in range(0, batch):
            dataX[i, j, :, :] = dataset[i:(i+look_back), j]
            dataY[i, j]       = dataset[i + look_back, j]
    return dataX, dataY

# def create_BatchD_dataset(dataset : np.ndarray, look_back : int = 1
#                     ) -> tuple[np.ndarray, np.ndarray]:
    
#     d_len : int = dataset.shape[0]-look_back
#     batch : int = dataset.shape[1]

#     dataX : np.ndarray = np.zeros(shape=(d_len*batch, look_back, 1),
#                                   dtype=np.float32)
#     dataY : np.ndarray = np.zeros(shape=(d_len*batch),
#                                   dtype=np.float32)
#     for i in range(0, d_len):
#         for j in range(0, batch):
#             dataX[i*batch:i*batch+j, :, :] = dataset[i:(i+look_back), j]
#             dataY[i*batch:i*batch+j]       = dataset[i + look_back, j]
#     return dataX, dataY

b_dataframe = scaler.fit_transform(b_dataframe)

trX, trY = create_Batch_dataset(b_dataframe, look_back)
trX = np.reshape(trX[:],newshape=(trX.shape[0]*trX.shape[1],1,1),order='C')
trY = np.reshape(trY[:],newshape=(trY.shape[0]*trY.shape[1],1),order='C')
# # %%
# @tf.autograph.experimental.do_not_convert
# def split_window(features : tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
#     inputs : tf.Tensor=features[:,input_slice,:-len(label_columns)]
    
#     labels : tf.Tensor = features[:, labels_slice, :]
#     labels = tf.stack(
#                 [labels[:, :, -i]
#                     for i in range(len(label_columns),0,-1)], axis=-1)
    
#     # Slicing doesn't preserve static shape information, so set the shapes
#     # manually. This way the `tf.data.Datasets` are easier to inspect.
#     inputs.set_shape([None, input_width, None])
#     labels.set_shape([None, label_width, None])
    
#     return inputs, labels

# data_ds : tf.raw_ops.BatchDataset = \
#     tf.keras.preprocessing.timeseries_dataset_from_array(
#         data=dataset, targets=None,
#         sequence_length=1, sequence_stride=1,
#         sampling_rate=1,
#         batch_size=12, shuffle=False,
#         seed=None, start_index=None, end_index=None
#     )

# data_ds : tf.raw_ops.MapDataset = data_ds.map(split_window)
# %%
h_nodes = 64
batch_size = trX.shape[1]
# model = tf.keras.models.Sequential([
#     tf.keras.layers.InputLayer(batch_input_shape=(1, 12, 1, 1)),
#     #tf.keras.layers.InputLayer(input_shape=(look_back, 1)),
#     tf.keras.layers.TimeDistributed(
#         tf.keras.layers.LSTM(units=h_nodes, 
#                         stateful=True, return_sequences=True)
#                 ),
#     tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
#     tf.keras.layers.TimeDistributed(
#         tf.keras.layers.LSTM(units=h_nodes, 
#                         stateful=True, return_sequences=True)
#                     ),
#     tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
#     tf.keras.layers.TimeDistributed(
#         tf.keras.layers.LSTM(units=int(h_nodes/2), 
#                         stateful=True, return_sequences=True)
#                     ),
#     tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
#     tf.keras.layers.TimeDistributed(
#         tf.keras.layers.LSTM(units=int(h_nodes/2), 
#                         stateful=True, return_sequences=False)
#                     ),
#     tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
#     tf.keras.layers.TimeDistributed(
#         tf.keras.layers.Dense(units=1, activation=None)
#                     )
# ])
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(batch_input_shape=(12, 1, 1)),
    tf.keras.layers.LSTM(units=h_nodes, 
                    stateful=True, return_sequences=True),    
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),    
    tf.keras.layers.LSTM(units=h_nodes, 
                    stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),    
    tf.keras.layers.LSTM(units=int(h_nodes/2), 
                    stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(units=int(h_nodes/2), 
                        stateful=True, return_sequences=False),    
    tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=1, activation=None)    
])
def custom_loss(y_true, y_pred):
    loss = tf.keras.backend.abs(y_pred - y_true)  # (batch_size, 2)
    tf.print(y_true)
    return loss
print(model.summary())
#model.fit(trX, trY, batch_size=12, epochs = 1, shuffle=False)
model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=[tf.metrics.MeanAbsoluteError(),
                       tf.metrics.RootMeanSquaredError()]
            )
#print(model.summary())
epochs : int = 1000
history = []
for i in range(1,epochs+1):
    print(f'Epoch {i}/{epochs}')
    # for j in range(0,trX.shape[0]):
    #     model.train_on_batch(trX[j,:,:,:], trY[j,:])
    history.append(model.fit(trX, trY, batch_size=12, epochs = 1, shuffle=False))
    #! Wont work. Needs a Callback for that.
    # if(i % train_df.shape[0] == 0):
    #     print("Reseting model")
    model.reset_states()
# %%
# for j in range(0,trX.shape[0]):
#     model.evaluate(trX[j,:,:,:], trY[j,:], batch_size=12, verbose=1)
model.evaluate(trX[:,:,:], trY[:,:], batch_size=12, verbose=1)
model.reset_states()

new_model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(batch_input_shape=(1, look_back, 1)),
    tf.keras.layers.LSTM(units=h_nodes, stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(units=h_nodes, stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(units=int(h_nodes/2), stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(units=int(h_nodes/2), stateful=True, return_sequences=False),
    tf.keras.layers.Dropout(rate=0.2, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=1, activation=None)
])
new_model.set_weights(model.get_weights())
new_model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=[tf.metrics.MeanAbsoluteError(),
                       tf.metrics.RootMeanSquaredError()]
            )
new_model.evaluate(trX[::12,:,:], trY[::12,], batch_size=1, verbose=1)

look_ahead = 250
xval = trX[-1:,:,:]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = triple_model.predict(xval[-1:,:,:], batch_size=1)
    predictions[i] = prediction
    xval = np.expand_dims(prediction, axis=1)

plt.figure(figsize=(12,5))
# plt.plot(np.arange(len(trainX)),np.squeeze(trainX))
# plt.plot(np.arange(200),scaler.inverse_transform(np.squeeze(trainPredict)[:,None][1:]))
# plt.plot(np.arange(200),scaler.inverse_transform(np.squeeze(testY)[:,None][:200]),'r')
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
#plt.plot(np.arange(look_ahead),dataset[train_size:(train_size+look_ahead)],
            #label="test function")
plt.legend()
plt.show()