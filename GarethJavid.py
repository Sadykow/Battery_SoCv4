#!/usr/bin/python
# %% [markdown]
# # # 1
# # #
# # LSTM for SoC by Ephrem Chemali 2017
# In this paper, we showcase how recurrent neural networks with an LSTM cell, a 
# machine learning technique, can accu-ately estimate SOC and can do so by
# self-learning the network parameters.

# Typical dataset used to train the networks is given by
#  D = {(Ψ 1 , SOC 1 ∗ ), (Ψ 2 , SOC 2 ∗ ), ..., (Ψ N , SOC N)}
# The vector of inputs is defined as 
# Ψ k = [V (k), I(k), T (k)], where V (k), I(k), T (k)

#? A final fully-connected layer performs a linear transformation on the 
#?hidden state tensor h k to obtain a single estimated SOC value at time step k.
#?This is done as follows: SOC k = V out h k + b y ,

#* Loss: sumN(0.5*(SoC-SoC*)^2)
#* Adam optimization: decay rates = b1=0.9 b2=0.999, alpha=10e-4, ?epsilon=10e-8

#* Drive cycles: over 100,000 time steps
#*  1 batch at a time.
#* Erors metricks: MAE, RMS, STDDEV, MAX error

# Data between 0 10 25C for training. Apperently mostly discharge cycle.
# Drive Cycles which included HWFET, UDDS, LA92 and US06
#* The network is trained on up to 8 mixed drive cycles while validation is
#*performed on 2 discharge test cases. In addition, a third test case, called
#*the Charging Test Case, which includes a charging profile is used to validate
#*the networks performance during charging scenarios.

#* LSTM-RNN with 500 nodes. Took 4 hours.
#* The MAE achieved on each of the first two test cases is 
#*0.807% and 1.252% (0.00807 and 0.01252)
#*15000 epoch.

#? This file used to train only on a FUDS dataset and validate against another
#?excluded FUDS datasets. In this case, out of 12 - 2 for validation.
#?Approximately 15~20% of provided data.
#? Testing performed on any datasets.
# %%
import os                       # OS, SYS, argc functions
import pandas as pd             # File read
import matplotlib as mpl        # Plot functionality
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import keras         # Tensorflow and Numpy replacement
import tensorflow_addons as tfa
import numpy as np
import logging

from sys import platform        # Get type of OS

from extractor.WindowGenerator import WindowGenerator
from extractor.DataGenerator import *
#from modules.RobustAdam import RobustAdam
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
    logging.info("GPU found and memory growth enabled") 
    
    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    logging.info("GPU found") 
    logging.debug(f"\nPhysical GPUs: {len(physical_devices)}"
                  f"\nLogical GPUs: {len(logical_devices)}")
#! For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float32')
# %%
profile : str = 'DST'
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
# data_soc = dataGenerator.train_SoC['SoC(%)']
# plt.scatter(range(0, data_soc.size),data_soc)
val_soc = dataGenerator.valid_SoC
plt.scatter(range(0, val_soc.size),val_soc)
# %%
window = WindowGenerator(Data=dataGenerator,
                        input_width=500, label_width=1, shift=0,
                        input_columns=['Current(A)', 'Voltage(V)', 'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,
                        includeTarget=False, normaliseLabal=False,
                        shuffleTraining=False)
ds_train, xx_train, yy_train = window.train
ds_valid, xx_valid, yy_valid = window.valid
#_, _, xx_train, yy_train = window.full_train
#_, _, xx_valid, yy_valid = window.full_valid
x_train = np.array(xx_train, copy=True, dtype=np.float32)
x_valid = np.array(xx_valid, copy=True, dtype=np.float32)
y_train = np.array(yy_train, copy=True, dtype=np.float32)
y_valid = np.array(yy_valid, copy=True, dtype=np.float32)
#train_df, train_ds, train_x, train_y = window.ParseFullData(dataGenerator.train_dir)
#plt.scatter(range(0, y_train.size),y_train)
#plt.scatter(range(0, y_valid.size),y_valid)
# %%
# window = WindowGenerator(Data=dataGenerator,
#                         input_width=1, label_width=1, shift=1,
#                         input_columns=['Current(A)', 'Voltage(V)', 'Temperature (C)_1'],
#                         label_columns=['SoC(%)'], batch=1,
#                         includeTarget=False, normaliseLabal=False,
#                         shuffleTraining=False)
# x_train, y_train = window.train
# x_valid, y_valid = window.valid
# %%
def custom_loss(y_true, y_pred):
    # #! No custom loss used in this implementation
    # #!Used standard mean_squared_error()
    # y_pred = tf.framework.ops.convert_to_tensor_v2_with_dispatch(y_pred)
    # y_true = tf.framework.ops.math_ops.cast(y_true, y_pred.dtype)
    # return tf.keras.backend.mean(tf.ops.math_ops.squared_difference(y_pred, y_true), axis=-1)
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))

model_loc : str = f'Models/GarethJavid2020/{profile}-models/'
iEpoch = 0
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
        tf.keras.layers.InputLayer(input_shape=x_train.shape[-2:],batch_size=None),
        tf.keras.layers.GRU(    #?260 by BinXia, times by 2 or 3
            units=500, activation='tanh', recurrent_activation='sigmoid',
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
                              activation=None) #! FIX THAT
    ])

checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath =model_loc+f'{profile}-checkpoints/checkpoint',
    monitor='val_loss', verbose=0,
    save_best_only=False, save_weights_only=False,
    mode='auto', save_freq='epoch', options=None,
)
# %%
# def adam(cost, params, alpha=np.float32(0.001), beta_1=np.float32(0.9), beta_2=np.float32(0.999), eps=np.float32(1e-8)):
#     g_params = tf.gradients(cost, params)
#     t = tf.Variable(0.0, dtype=tf.float32, name='t')
#     updates = []
#     updates.append(t.assign(t + 1))
#     with tf.control_dependencies(updates):
#         for param, g_param in zip(params, g_params):
#             m = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='m')
#             v = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='v')
#             alpha_t = alpha*tf.sqrt(1. - beta_2**t)/(1. - beta_1**t)
#             updates.append(m.assign(beta_1*m + (1. - beta_1)*g_param))
#             with tf.control_dependencies(updates):
#                 updates.append(v.assign(beta_2*v + (1. - beta_2)*g_param**2))
#             with tf.control_dependencies(updates):
#                 updates.append(param.assign(param - alpha_t*m/(tf.sqrt(v) + eps)))
#     return updates

# x = tf.placeholder(tf.float32, [None, 784])
# t = tf.placeholder(tf.float32, [None, 10])

# layers = [
#     Dense(784, 256, tf.nn.relu),
#     Dense(256, 256, tf.nn.relu),
#     Dense(256, 10, tf.nn.softmax)
# ]

# y, params = f_props(layers, x)

# cost = -tf.reduce_mean(tf.reduce_sum(t*tf_log(y), axis=1))

# # Choose an optimizer from sgd, sgd_clip, momentum, nesterov_momentum, adagrad, adadelta, rmsprop, adam, adamax, smorms3
# updates = adam(cost, params)

# train = tf.group(*updates)
# test  = tf.argmax(y, axis=1)

# n_epochs = 10
# batch_size = 100
# n_batches = train_X.shape[0]//batch_size

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(n_epochs):
#         train_X, train_y = shuffle(train_X, train_y, random_state=random_state)
#         for i in range(n_batches):
#             start = i*batch_size
#             end = start + batch_size
#             sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})
#         pred_y, valid_cost = sess.run([test, cost], feed_dict={x: valid_X, t: valid_y})
#         print('EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y, average='macro')))

# %%
class SGOptimizer(tf.keras.optimizers.Optimizer):
    """Gradient Descent
        New_weight = weight  - eta * rate of change of error wrt weight
        w = w - η* ∂E/∂w 

    Args:
        tf ([type]): [description]
    """
    def __init__(self, learning_rate=0.01, name="SGOptimizer", **kwargs):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate)) # handle lr=learning_rate
        self._is_first = True
    
    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "pv") #previous variable i.e. weight or bias
        for var in var_list:
            self.add_slot(var, "pg") #previous gradient


    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform one optimization step for one model variable
        """
        # def adam(cost, params, alpha=np.float32(0.001),
        #          beta_1=np.float32(0.9), beta_2=np.float32(0.999),
        #          eps=np.float32(1e-8)):
        #     g_params = tf.gradients(cost, params)
        #     t = tf.Variable(0.0, dtype=tf.float32, name='t')
        #     updates = []
        #     updates.append(t.assign(t + 1))
        #     with tf.control_dependencies(updates):
        #         for param, g_param in zip(params, g_params):
        #             m = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='m')
        #             v = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='v')
        #             alpha_t = alpha*tf.sqrt(1. - beta_2**t)/(1. - beta_1**t)
        #             updates.append(m.assign(beta_1*m + (1. - beta_1)*g_param))
        #             with tf.control_dependencies(updates):
        #                 updates.append(v.assign(beta_2*v + (1. - beta_2)*g_param**2))
        #             with tf.control_dependencies(updates):
        #                 updates.append(param.assign(param - alpha_t*m/(tf.sqrt(v) + eps)))
        #     return updates

        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype) # handle learning rate decay
        new_var_m = var - grad * lr_t
        pv_var = self.get_slot(var, "pv")
        pg_var = self.get_slot(var, "pg")
        
        if self._is_first:
            print("First Pass")
            self._is_first = False
            new_var = new_var_m
            tf.print(new_var)
            print("=========\n\n")
        else:
            print("Second Pass")
            cond = grad*pg_var >= 0
            tf.print(cond)
            avg_weights = (pv_var + var)/2.0
            new_var = tf.where(cond, new_var_m, avg_weights)
            print("=========\n\n")
        pv_var.assign(var)
        pg_var.assign(grad)
        var.assign(new_var)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError
    
    def update_labels(self, loss_value : np.float32, label : np.float32):
        #print('Updating losses')
        self.prediction = tf.Variable(loss_value, dtype=np.float32)
        self.actual = tf.Variable(label, dtype=np.float32)
        #print(self.prediction, self.actual)
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
        }

#from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.ops import state_ops
#@keras_export('tf.keras.optimizers.RobustAdam')
class RobustAdam(tf.keras.optimizers.Optimizer):
    
    epsilon : tf.float32 = 10e-8
    _is_first : bool = True
    first_run : bool = True

    # Robust part
    cost_func = None
    prediction : tf.Variable = 0.0
    actual : tf.Variable = 0.0
    prev_loss : tf.Variable = 0.0
    
    prev_var_1 : tf.Variable = 0.0
    prev_var_2 : tf.Variable = 0.0
    prev_var_3 : tf.Variable = 0.0
    prev_var_4 : tf.Variable = 0.0
    prev_var_5 : tf.Variable = 0.0

    i_init : int = 0
    t : int = 0

    def __init__(self, name : str ='RobustAdam',
                 lr_rate: tf.float32 = 0.001,
                 beta_1 : tf.float32 = 0.9,
                 beta_2 : tf.float32 = 0.999,
                 beta_3 : tf.float32 = 0.999,
                 epsilon :tf.float32 = 1e-7,
                 cost = None,
                 **kwargs):
        #super().__init__(name, **kwargs)
        super(RobustAdam, self).__init__(name, **kwargs)
        # handle lr=learning_rate
        self._set_hyper("learning_rate", kwargs.get("lr", lr_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('beta_3', beta_3)
        self.epsilon = epsilon # or backend_config.epsilon()
        self._is_first = True
        self.cost_func = cost
                
    
    def _create_slots(self, var_list : list[tf.Variable]):
        """ For each model variable, create the optimizer variable associated
        with it. TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.

        Create slots for the first, second and third moments.
        Separate for-loops to respect the ordering of slot variables from v1.

        Args:
            var_list ([type]): [description]

        Returns:
            [type]: [description]
        """
        #print("Var_List: {}\n\n VLType: {}".format(var_list, type(var_list[0])))
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        for var in var_list:
            self.add_slot(var, 'd')
    
    def _prepare_local(self, var_device, var_dtype, apply_state):
        #print()
        super(RobustAdam, self)._prepare_local(var_device, var_dtype, apply_state)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        beta_3_t = self._get_hyper('beta_3', var_dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
            (tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,                
                epsilon=np.float32(self.epsilon),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t,
                beta_3_t=beta_3_t,
                one_minus_beta_3_t=1 - beta_3_t
                ))

    #@tf.function
    def _resource_apply_dense(self, grad, var, apply_state=None):
        """ Update the slots and perform one optimization step for one model
        variable.

        Args:
            grad ([type]): [description]
            var ([type]): [description]
            apply_state ([type]): [description]

        Returns:
            [type]: [description]
        """
        var_device, var_dtype = var.device, var.dtype.base_dtype        
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
        m = self.get_slot(var=var, slot_name='m')
        v = self.get_slot(var=var, slot_name='v')
        d = self.get_slot(var=var, slot_name='d')
        
        # m_t = beta1 * m + (1 - beta1) * g_t
        m_scaled_g_values = coefficients['one_minus_beta_1_t'] * grad
        m_t = state_ops.assign(ref=m, value=m * coefficients['beta_1_t'],
                        use_locking=self._use_locking,
                        validate_shape=None, name='m_t')
        with tf.control_dependencies([m_t]):
            # m_t = m_t.assign_add(m_scaled_g_values)
            m_t = state_ops.assign_add(ref=m_t, value=m_scaled_g_values,
                        use_locking=self._use_locking, name=None)
            # m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
        
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v_scaled_g_values = coefficients['one_minus_beta_2_t']*(grad * grad)
        v_t = state_ops.assign(ref=v, value=v * coefficients['beta_2_t'],
                        use_locking=self._use_locking,
                        validate_shape=None, name='v_t')
        with tf.control_dependencies([v_t]):
            #v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
            #     v.assign(beta_2*v + (1. - beta_2)*g_param**2)
            #! Use state_ops
            v_t = state_ops.assign_add(ref=v_t, value=v_scaled_g_values,
                        use_locking=self._use_locking, name=None)
            #v_t = v_t.assign_add(v_scaled_g_values)
        
        if self.first_run:
            self.prev_loss = self.cost_func(y_true=self.actual,
                                            y_pred=self.prediction)
            # tf.print(self.actual)
            # tf.print(self.prediction)
            # tf.print(self.prev_loss)
            # print("=========\n\n")
            var_update = state_ops.assign_sub(
                ref=var, 
                value=coefficients['lr'] * m_t /\
                    (tf.sqrt(v_t) + coefficients['epsilon']),
                use_locking=self._use_locking)
            
            if self.i_init == 0:
                self.prev_var_1 = var_update
            elif self.i_init == 1:
                self.prev_var_2 = var_update
            elif self.i_init == 2:
                self.prev_var_3 = var_update
            elif self.i_init == 3:
                self.prev_var_4 = var_update
            else:
                self.prev_var_5 = var_update
            
            self.i_init += 1
            
            return [var_update, m_t, v_t]
        
            # cond = grad*pg_var >= 0
            # print(cond)
            # avg_weights = (pv_var + var)/2.0
            # new_var = tf.where(cond, new_var_m, avg_weights)
        else :
            #self._transform_loss()
            # tf.print(grad)
            #print(coefficients)
            #print(self._transform_loss())
            loss = lambda var : tf.sqrt(tf.reduce_mean(tf.square(var), axis=0))
            # print(tf.reduce_mean(self.prev_var, axis=0))

            # print(tf.compat.v1.losses.compute_weighted_loss(
            #         (var)))
            # print('\n\n')
            # self.prev_var = var
            current_loss = self.cost_func(y_true=self.actual,
                                        y_pred=self.prediction)
            # tf.print(self.actual)
            # tf.print(self.prediction)
            # tf.print(current_loss)
            # tf.print(self.prev_loss)
            
            if self.i_init % 5 == 0:
                r_t = tf.abs(x=tf.divide(
                        x=var,
                        y=self.prev_var_1,
                        name='Robust'
                    )
                )
                self.prev_var_1 = var
            elif self.i_init % 5 == 1:
                r_t = tf.abs(x=tf.divide(
                        x=var,
                        y=self.prev_var_2,
                        name='Robust'
                    )
                )
                self.prev_var_2 = var
            elif self.i_init  % 5 == 2:
                r_t = tf.abs(x=tf.divide(
                        x=var,
                        y=self.prev_var_3,
                        name='Robust'
                    )
                )
                self.prev_var_3 = var
            elif self.i_init  % 5 == 3:
                r_t = tf.abs(x=tf.divide(
                        x=var,
                        y=self.prev_var_4,
                        name='Robust'
                    )
                )
                self.prev_var_4 = var
            else:
                r_t = tf.abs(x=tf.divide(
                        x=var,
                        y=self.prev_var_5,
                        name='Robust'
                    )
                )
                self.prev_var_5 = var
            self.i_init += 1
            #r_t = current_loss / self.prev_loss
            self.prev_loss = current_loss
            #print(r_t)
            # if current_loss < self.prev_loss:
            #     print("True1")
            # else:
            #     print("False2")
            # # print(var[0])
            # # r_t = tf.divide(x=var,
            # #                 y=self.prev_var,
            # #                 name='Robust')
            
            d_scaled_L = coefficients['one_minus_beta_3_t']*(r_t)
            # print(d)
            # print(coefficients['one_minus_beta_3_t'])
            # print(d_scaled_L)
            d_t = state_ops.assign(ref=d, value=d*coefficients['beta_3_t'],
                            use_locking=self._use_locking,
                            validate_shape=None, name='d_t')
            # print(d_t)
            with tf.control_dependencies([d_t]):
                # d_t = d_t.assign_add(d_scaled_L)
                d_t = state_ops.assign_add(ref=d_t, value=d_scaled_L,
                            use_locking=self._use_locking, name=None)
                    
            var_update = state_ops.assign_sub(
                ref=var, 
                value=coefficients['lr'] * m_t /(d_t*tf.sqrt(v_t) + coefficients['epsilon']),
                use_locking=self._use_locking)
            
            # var_update = state_ops.assign_sub(
            #         ref=var, 
            #         value=coefficients['lr'] * m_t /\
            #             (tf.sqrt(v_t) + coefficients['epsilon']),
            #         use_locking=self._use_locking)
            return [var_update, m_t, v_t]
        # # d_t = beta3 * d + (1 - beta3) * r_t
        # var_dtype = var.dtype.base_dtype
        # lr_t = self._decayed_lr(var_dtype) # handle learning rate decay
        # new_var_m = var - grad * lr_t
        # pv_var = self.get_slot(var, "v")
        # pg_var = self.get_slot(var, "m")
        
        # if self._is_first:
        #     self._is_first = False
        #     new_var = new_var_m
        # else:
        #     cond = grad*pg_var >= 0
        #     print(cond)
        #     avg_weights = (pv_var + var)/2.0
        #     new_var = tf.where(cond, new_var_m, avg_weights)
        # pv_var.assign(var)
        # pg_var.assign(grad)
        # var.assign(new_var)

        #return super()._resource_apply_dense(grad, handle, apply_state)
    def update_labels(self, loss_value : np.float32, label : np.float32,
                      first_run : bool = True, t : int = 0):
        #print('Updating losses')
        self.prediction = tf.Variable(loss_value, dtype=np.float32)
        self.actual = tf.Variable(label, dtype=np.float32)
        self.first_run = first_run
        self.t = t
        #print(self.prediction, self.actual)

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        # return super()._resource_apply_sparse(grad, handle,
        #                                       indices, apply_state)
        raise NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay'        : self._serialize_hyperparameter('decay'),
            'beta_1'       : self._serialize_hyperparameter('beta_1'),
            'beta_2'       : self._serialize_hyperparameter('beta_2'),
            'beta_3'       : self._serialize_hyperparameter('beta_3'),
            'epsilon'      : self.epsilon,
        })
        return config

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# def loss(model, x, y, training):
#   # training=training is needed only if there are layers with different
#   # behavior during training versus inference (e.g. Dropout).
#   y_ = model(x, training=training)
#   return loss_object(y_true=y, y_pred=y_)
# l = loss(gru_model, x_train[:500,:,:], y_train[:500], training=False)
# print("Loss test: {}".format(l))

# gru_model.compile(loss=custom_loss,
#             optimizer=RobustAdam(learning_rate = 0.001,
#                  beta_1 = 0.9, beta_2 = 0.999, beta_3 = 0.999, epsilon = 1e-7,
#                  cost = custom_loss),
#             metrics=[tf.metrics.MeanAbsoluteError(),
#                      tf.metrics.RootMeanSquaredError(),
#                      tfa.metrics.RSquare(y_shape=(1,), dtype=tf.float32)],
#             #run_eagerly=True
#             )
# history = gru_model.fit(x=x_train[:1000,:,:], y=y_train[:1000], epochs=3,
#                     validation_data=(x_valid[:200,:,:], y_valid[:200]),
#                     callbacks=None, batch_size=1, shuffle=True
#                     )
loss_object = tf.keras.losses.MeanSquaredError()
def custom_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))

def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    #return loss_object(y_true=y, y_pred=y_)
    return custom_loss(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
        grads = tape.gradient(loss_value, model.trainable_variables)    
    return loss_value, grads

#optimiser = SGOptimizer(learning_rate=0.001)
optimiser = RobustAdam(lr_rate = 0.001,
                 beta_1 = 0.9, beta_2 = 0.999, beta_3 = 0.999, epsilon = 1e-7,
                 cost = custom_loss)

first_run = True
size = 10
t = 0
for x, y in zip(np.expand_dims(x_train[:size,:,:], axis=1), y_train[:size]):
    with tf.GradientTape() as tape:
        #current_loss = custom_loss(gru_model(x_train[:1,:,:]))
        loss_value = loss(gru_model, x, y, training=True)
        grads = tape.gradient(loss_value, gru_model.trainable_variables)
    print(f'LossValue: {loss_value}, True {y[0]}')
    #! First pass bool here.
    if first_run:
        optimiser.update_labels(loss_value = loss_value, label=y[0],
                                first_run = first_run, t=t)
        first_run = False
    else:
        optimiser.update_labels(loss_value = loss_value, label=y[0],
                                first_run = first_run, t=t)
    optimiser.apply_gradients(zip(grads, gru_model.trainable_variables))
    t += 1
# for x, y in zip(np.expand_dims(x_train[:100,:,:], axis=1), y_train[0:100]):
#     loss_value, grads = grad(gru_model, x, y)    
#     optimiser.apply_gradients(zip(grads, gru_model.trainable_variables))

# l = loss(gru_model, x_train[0:1,:,:], y_train[0:1], training=False)
# print("Loss test: {}".format(l))


# %%

mEpoch : int = 40
firtstEpoch : bool = True
while iEpoch < mEpoch:
    iEpoch+=1
    print(f"Epoch {iEpoch}/{mEpoch}")
    
    history = gru_model.fit(x=x_train, y=y_train, epochs=1,
                        validation_data=(x_valid, y_valid),
                        callbacks=[checkpoints], batch_size=1, shuffle=True
                        )#! Initially Batch size 1; 8 is safe to run - 137s
    # history = lstm_model.fit(x=ds_train, epochs=1,
    #                     validation_data=ds_valid,
    #                     callbacks=[checkpoints], batch_size=1
    #                     )#! Initially Batch size 1; 8 is safe to run - 137s
    gru_model.save(f'{model_loc}{iEpoch}')
    gru_model.save_weights(f'{model_loc}weights/{iEpoch}')
    
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

