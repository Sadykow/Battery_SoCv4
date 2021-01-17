# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#from IPython import get_ipython

# %%
import numpy as np
#from standard_plots import *
import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

#import theano

#get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# # LSTMs

# %%
dataset = np.cos(np.arange(1000)*(20*np.pi/1000))[:,None]
plt.plot(dataset)


# %%
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# %% [markdown]
# ## Window of 20 time steps

# %%
look_back = 20
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


# %%
train.shape


# %%
trainX.shape


# %%
trainY.shape


# %%
#get_ipython().run_cell_magic('time', '', 'theano.config.compute_test_value = "ignore"\n# create and fit the LSTM network\nbatch_size = 1\nmodel = Sequential()\nmodel.add(LSTM(32,input_dim=1))\nmodel.add(Dropout(0.3))\nmodel.add(Dense(1))\nmodel.compile(loss=\'mean_squared_error\', optimizer=\'adam\')\nmodel.fit(trainX, trainY, nb_epoch=100, batch_size=batch_size, verbose=2)')
batch_size = 1
model = tf.keras.models.Sequential([
    #tf.keras.layers.InputLayer(batch_input_shape=(1, 1, 1)),
    tf.keras.layers.LSTM(32,input_dim=1),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),    
    tf.keras.layers.Dense(units=1,
                            activation=None)
])
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=batch_size, verbose=2)
# %%
trainScore = model.evaluate(trainX, trainY, batch_size=batch_size, verbose=0)
print('Train Score: ', trainScore)
testScore = model.evaluate(testX[:252], testY[:252],
                            batch_size=batch_size, verbose=0)
print('Test Score: ', testScore)
# %%
look_ahead = 250
trainPredict = [np.vstack([trainX[-1][1:], trainY[-1]])]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = model.predict(np.array([trainPredict[-1]]),
                                batch_size=batch_size)
    predictions[i] = prediction
    trainPredict.append(np.vstack([trainPredict[-1][1:],prediction]))


# %%
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
# ## Stateful LSTMs

# %%
look_back = 20
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


# %%
#get_ipython().run_cell_magic('time', '', 'theano.config.compute_test_value = "ignore"\n# create and fit the LSTM network\nbatch_size = 1\nmodel = Sequential()\n# model.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))\n# model.add(Dropout(0.3))\n# model.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))\n# model.add(Dropout(0.3))\nmodel.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True))\nmodel.add(Dropout(0.3))\nmodel.add(Dense(1))\nmodel.compile(loss=\'mean_squared_error\', optimizer=\'adam\')\nfor i in range(200):\n    model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=0, shuffle=False)\n    model.reset_states()')

batch_size = 1
model = tf.keras.models.Sequential([
    #tf.keras.layers.InputLayer(batch_input_shape=(1, 1, 1)),
    tf.keras.layers.LSTM(32, batch_input_shape=(batch_size, look_back, 1),
                         stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(32, batch_input_shape=(batch_size, look_back, 1),
                        stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(32, batch_input_shape=(batch_size, look_back, 1),
                        stateful=True),
    tf.keras.layers.Dense(units=1,
                            activation=None)
])
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(200):
    model.fit(trainX, trainY, epochs=1, batch_size=batch_size,
                verbose=0, shuffle=False)
    model.reset_states()

# %%
trainScore = model.evaluate(trainX, trainY, batch_size=batch_size, verbose=0)
print('Train Score: ', trainScore)
testScore = model.evaluate(testX[:252], testY[:252], batch_size=batch_size,
                            verbose=0)
print('Test Score: ', testScore)
# %%
look_ahead = 250
trainPredict = [np.vstack([trainX[-1][1:], trainY[-1]])]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = model.predict(np.array([trainPredict[-1]]),
                                batch_size=batch_size)
    predictions[i] = prediction
    trainPredict.append(np.vstack([trainPredict[-1][1:],prediction]))
# %%
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
# ## Stateful LSTMs with wider window

# %%
look_back = 40
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


# %%
#get_ipython().run_cell_magic('time', '', 'theano.config.compute_test_value = "ignore"\n# create and fit the LSTM network\nbatch_size = 1\nmodel = Sequential()\n# model.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))\n# model.add(Dropout(0.3))\n# model.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))\n# model.add(Dropout(0.3))\nmodel.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True))\nmodel.add(Dropout(0.3))\nmodel.add(Dense(1))\nmodel.compile(loss=\'mean_squared_error\', optimizer=\'adam\')\nfor i in range(200):\n    model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=0, shuffle=False)\n    model.reset_states()')
batch_size = 1
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, batch_input_shape=(batch_size, look_back, 1),
                         stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(32, batch_input_shape=(batch_size, look_back, 1),
                        stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(32, batch_input_shape=(batch_size, look_back, 1),
                        stateful=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=1,
                            activation=None)
])
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(200):
    model.fit(trainX, trainY, epochs=1, batch_size=batch_size,
                verbose=1, shuffle=False)
    model.reset_states()
# %%
trainScore = model.evaluate(trainX, trainY, batch_size=batch_size, verbose=0)
print('Train Score: ', trainScore)
testScore = model.evaluate(testX[:252], testY[:252], batch_size=batch_size,
                            verbose=0)
print('Test Score: ', testScore)
# %%
look_ahead = 250
trainPredict = [np.vstack([trainX[-1][1:], trainY[-1]])]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = model.predict(np.array([trainPredict[-1]]),
                                batch_size=batch_size)
    predictions[i] = prediction
    trainPredict.append(np.vstack([trainPredict[-1][1:],prediction]))


# %%
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
# ## Stateful LSTMs, Stacked

# %%
batch_size = 1
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, batch_input_shape=(batch_size, look_back, 1),
                         stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(32, batch_input_shape=(batch_size, look_back, 1),
                        stateful=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=1,
                            activation=None)
])
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(200):
    model.fit(trainX, trainY, epochs=1, batch_size=batch_size,
                verbose=0, shuffle=False)
    model.reset_states()
# %%
trainScore = model.evaluate(trainX, trainY, batch_size=batch_size, verbose=0)
print('Train Score: ', trainScore)
testScore = model.evaluate(testX[:252], testY[:252], batch_size=batch_size,
                            verbose=0)
print('Test Score: ', testScore)
# %%
look_ahead = 250
trainPredict = [np.vstack([trainX[-1][1:], trainY[-1]])]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = model.predict(np.array([trainPredict[-1]]),
                                batch_size=batch_size)
    predictions[i] = prediction
    trainPredict.append(np.vstack([trainPredict[-1][1:],prediction]))
# %%
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
#get_ipython().run_cell_magic('time', '', 'theano.config.compute_test_value = "ignore"\n# create and fit the LSTM network\nbatch_size = 1\nmodel = Sequential()\nfor i in range(2):\n    model.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))\n    model.add(Dropout(0.3))\nmodel.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True))\nmodel.add(Dropout(0.3))\nmodel.add(Dense(1))\nmodel.compile(loss=\'mean_squared_error\', optimizer=\'adam\')\nfor i in range(200):\n    model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=0, shuffle=False)\n    model.reset_states()')
batch_size = 1
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, batch_input_shape=(batch_size, look_back, 1),
                         stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(32, batch_input_shape=(batch_size, look_back, 1),
                        stateful=True, return_sequences=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.LSTM(32, batch_input_shape=(batch_size, look_back, 1),
                        stateful=True),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=1,
                            activation=None)
])
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(200):
    model.fit(trainX, trainY, epochs=1, batch_size=batch_size,
                verbose=0, shuffle=False)
    model.reset_states()
# %%
trainScore = model.evaluate(trainX, trainY, batch_size=batch_size, verbose=0)
print('Train Score: ', trainScore)
testScore = model.evaluate(testX[:252], testY[:252], batch_size=batch_size,
                            verbose=0)
print('Test Score: ', testScore)
# %%
look_ahead = 250
trainPredict = [np.vstack([trainX[-1][1:], trainY[-1]])]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = model.predict(np.array([trainPredict[-1]]),
                                batch_size=batch_size)
    predictions[i] = prediction
    trainPredict.append(np.vstack([trainPredict[-1][1:],prediction]))


# %%
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
# ## Normal Deep Learning in Keras

# %%
#get_ipython().run_cell_magic('time', '', 'trainX = np.squeeze(trainX)\ntestX = np.squeeze(testX)\ntheano.config.compute_test_value = "ignore"\n# create and fit the LSTM network\nbatch_size = 1\nmodel = Sequential()\nmodel.add(Dense(output_dim=32,input_dim=40,activation="relu"))\nmodel.add(Dropout(0.3))\nfor i in range(2):\n    model.add(Dense(output_dim=32,activation="relu"))\n    model.add(Dropout(0.3))\nmodel.add(Dense(1))\nmodel.compile(loss=\'mean_squared_error\', optimizer=\'adagrad\')\nmodel.fit(trainX, trainY, nb_epoch=100, batch_size=32, verbose=0)')
trainX = np.squeeze(trainX)
testX = np.squeeze(testX)
batch_size = 1
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=32,input_dim=40,activation="relu"),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=32,activation="relu"),
    tf.keras.layers.Dropout(rate=0.3, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=1,
                            activation=None)
])
model.compile(loss='mean_squared_error', optimizer='adagrad')
model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=0)
# %%
trainScore = model.evaluate(trainX, trainY, batch_size=batch_size, verbose=0)
print('Train Score: ', trainScore)
testScore = model.evaluate(testX[:252], testY[:252], batch_size=batch_size,
                            verbose=0)
print('Test Score: ', testScore)
# %%
look_ahead = 250
xval = np.hstack([trainX[-1][1:], trainY[-1]])[None,:]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = model.predict(xval, batch_size=32)
    predictions[i] = prediction
    xval = np.hstack([xval[:,1:],prediction])
# %%
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