# %% [markdown]
# # Window Generator
# Using data from Data Generator - creates Windows to use.
# Used to separate huge code ammounts to separate files.
# %%
import os
from time import perf_counter
from matplotlib.pyplot import axis
from numpy.core.fromnumeric import ndim
import pandas as pd             # Tabled data storage
import tensorflow as tf
from tensorflow.python.keras.backend import dtype
from tensorflow.python.lib.io.file_io import copy
import tensorflow.experimental.numpy as tnp
import numpy as np
# Used for making annotations
from dataclasses import dataclass

from extractor.DataGenerator import DataGenerator
from sklearn.preprocessing import MinMaxScaler
from extractor.soc_calc import diffSoC

from numba import jit, vectorize
class WindowGenerator():
  Data : DataGenerator          # Data object containing Parsed data

  input_width   : int           # Width of the input window
  label_width   : int           # Width of the output
  shift         : int           # Step or skip
  label_columns : list[str]     # List of targets
  input_columns : list[str]     # List of Input features
  batching      : bool          # Enable bacthning with default set
  batch         : int           # Bathes size. Samples simeltaniosly.
  includeTarget : bool          # Use Target as part of dataset.
  normaliseInput: bool          #
  normaliseLabal: bool          #
  
  float_dtype   : type          # Float variable Type
  int_dtype     : type          # Int variable Type

  shuffleTraining: bool         # Shuffeling Training data
  #label_columns_indices : list[int]
  #column_indices: list[int]
  total_window_size : int       # Size (input+shift)
  input_slice   : slice         # Input Slice (0:input_width)
  #input_indices : list[int]
  labels_slice  : slice
  #label_indices : list[int]

  def __init__(self, Data : DataGenerator,
               input_width : int, label_width : int, shift : int,
               input_columns : list[str], label_columns : list[str],
               batching : bool = False,
               batch : int = 1, includeTarget : bool = False,
               normaliseInput : bool = True, normaliseLabal : bool = True,
               shuffleTraining : bool = False,
               float_dtype : type = None,
               int_dtype : type = None) -> None:
    """ Window Constructor used to store data sets and properties of the 
        window, which will be used for processing.

    Args:
        Data (DataGenerator): Data Object containig prepared Train/Valid data
        input_width (int): Size of the input
        label_width (int): Size of the output
        shift (int): Step or shift in data window
        label_columns (list[str]): Label column names
        batching (bool): Enabeling batching mechanism with default list size
        batch (int, optional): Size of the batches define how many samples
    gets dead together. Defaults to 1.
        includeTarget (bool, optional): Apply target to main dataset. Target has
    to be the last column, for now. Defaults to False.
        float_dtype (type, optional): Standard float type for all objects. If
    None then gets type from DataGenerator object. Defaults to None.
        int_dtype (type, optional): Standard float type for all objects. If
    None then gets type from DataGenerator object. Defaults to None.
    """
    # Object containing Data from Excel
    self.Data = Data

    # Store raw data information
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift
    #self.label_names = label_names
    self.batching = batching
    self.batch = batch
    self.includeTarget = includeTarget
    self.normaliseInput = normaliseInput
    self.normaliseLabal = normaliseLabal
    self.shuffleTraining = shuffleTraining
    # Variable types to use
    if float_dtype:
      self.float_dtype = float_dtype
    else:
      #Get from Data generator
      self.float_dtype = self.Data.float_dtype

    if int_dtype:
      self.int_dtype = int_dtype
    else:
      self.int_dtype = self.Data.int_dtype
    
    # Work out the label column indices.
    self.input_columns = input_columns
    self.label_columns = label_columns
    # if self.label_columns is not None:
    #     self.label_columns_indices = {name: i for i, name in
    #                             enumerate(label_columns)}
    # overall_columns = self.Data.train[input_columns].columns
    # overall_columns = overall_columns.append(
    #                           self.Data.train_SoC[label_columns].columns)
    # self.column_indices = {name: i for i, name in
    #                     enumerate(overall_columns)}
    
    # Work out other parameters
    self.total_window_size = self.input_width + self.shift
    # Creating Slices
    #* Expected return similar to:
    #*input = 6, shift=1, label=1
    #*[0 1 2 3 4 5] & [6]
    self.input_slice = slice(0, self.input_width)    # [0:input_width]
    # self.input_indices = tnp.arange(start=0,
    #                     stop=self.total_window_size,
    #                     dtype=int_dtype)[self.input_slice]

    label_start : int = self.total_window_size - self.label_width
    self.labels_slice = slice(label_start, None) #[label_Start:]
    # self.label_indices = tnp.arange(start=0,
    #                     stop=self.total_window_size,
    #                     dtype=int_dtype)[self.labels_slice]

  def __repr__(self) -> str:
    """ A return from the constructor. Information of the storage like:
    Total windows size, input and label indices, Label/output column names.

    Returns:
        str: New line string with object information
    """
    return '\n'.join([
        f'\n\nTotal window size:        {self.total_window_size}',
        f'Input indices:\n{self.input_indices}',
        f'Label indices:        {self.label_indices}',
        f'Label column name(s): {self.label_columns}',
        #f'Label Column indices: {self.label_columns_indices}',
        #f'Column indices:       {self.column_indices}',
        f'Input Slice:          {self.input_slice}',
        #f'Input indices:        {self.input_indices}',
        f'Labels slice:         {self.labels_slice}',
        #f'Labels indices:       {self.label_indices}',
        '\n\n'])
  
  @tf.autograph.experimental.do_not_convert
  def make_dataset_from_array(self, inputs : np.ndarray,
                                    labels : np.ndarray
              ) -> tf.raw_ops.MapDataset:

    input_length : int = len(self.input_columns)    
    tic : float = perf_counter()    
    if self.normaliseInput: # Normalise Inputs
      MEAN = np.mean(a=self.Data.train[:,:input_length], axis=0,
                                      dtype=self.float_dtype,
                                      keepdims=False)
      STD = np.std(a=self.Data.train[:,:input_length], axis=0,
                                    keepdims=False)
      data : np.ndarray = np.divide(
                            np.subtract(
                                  np.copy(a=inputs[:,:input_length]),
                                  MEAN
                                ),
                            STD
                          )
    else:
      data : np.ndarray = np.copy(a=inputs[:,:input_length],
                                  order='K', subok=False)
    
    data = np.append(arr=data,
                      values=labels,
                      axis=1)
    ds : tf.raw_ops.BatchDataset = \
          tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data, targets=None,
            sequence_length=self.total_window_size, sequence_stride=1,
            sampling_rate=1,
            batch_size=1, shuffle=False,
            seed=None, start_index=None, end_index=None
        )

    ds : tf.raw_ops.MapDataset = ds.map(self.split_window)
    x : np.ndarray = np.asarray(list(ds.map(
                                lambda x, _: x[0,:,:]
                              ).as_numpy_iterator()
                          ))
    # print('Output Shape ')
    y : np.ndarray = np.asarray(list(ds.map(
                                lambda _, y: y[0,0]
                              ).as_numpy_iterator()
                          ))
    print(f"\n\nData windowing took: {(perf_counter() - tic):.2f} seconds")
    return ds, x, y

  @tf.autograph.experimental.do_not_convert
  def make_dataset_from_list(self, X : list[np.ndarray],
                                   Y : list[np.ndarray],
                                   look_back : int = 1
              ) -> tuple[np.ndarray, np.ndarray]:
    batch : int = len(X)
    dataX : list[np.ndarray] = []
    dataY : list[np.ndarray] = []
    
    input_length : int = len(self.input_columns)
    MEAN = np.mean(a=self.Data.train[:,:input_length], axis=0,
                                          dtype=self.float_dtype,
                                          keepdims=False)
    STD = np.std(a=self.Data.train[:,:input_length], axis=0,
                                   keepdims=False)
    tic : float = perf_counter()
    for i in range(0, batch):
        d_len : int = X[i].shape[0]-look_back
        dataX.append(np.zeros(shape=(d_len, look_back, input_length),
                    dtype=self.float_dtype))
        dataY.append(np.zeros(shape=(d_len,), dtype=self.float_dtype))
        for j in range(0, d_len):
            if self.normaliseInput: #! Spmething wrong here
              # dataX[i][j,:,:] = (X[i][self.input_columns].to_numpy()[j:(j+look_back), :]-MEAN)/STD
              dataX[i][j,:,:] =np.divide(
                                np.subtract(
                                      np.copy(a=X[i][self.input_columns].to_numpy()[j:(j+look_back), :]),
                                      MEAN
                                    ),
                                STD
                              )
            else:
              dataX[i][j,:,:] = X[i][self.input_columns].to_numpy()[j:(j+look_back), :]
            #dataY[i][j]     = Y[i][j+look_back,]
            dataY[i][j]     = Y[i][j+look_back-1,]

    print(f"\n\nData windowing took: {(perf_counter() - tic):.2f} seconds")
    return dataX, dataY

  @tf.autograph.experimental.do_not_convert
  def split_window(self, features : tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    if self.includeTarget:
      inputs : tf.Tensor=features[:,self.input_slice,:]
    else:
      #! Fix exlude similarly to tf.stack()
      inputs : tf.Tensor=features[:,self.input_slice,:-len(self.label_columns)]
    
    labels : tf.Tensor = features[:, self.labels_slice, :]
    #labels = tf.stack([labels[:, :, -2], labels[:, :, -1]], axis=-1)
    #labels = tf.stack([labels[:, :, -1]], axis=-1)
    labels = tf.stack(
                [labels[:, :, -i]
                    for i in range(len(self.label_columns),0,-1)], axis=-1)
    
    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    
    # if self.input_width == 1:
    #   return tf.transpose(
    #                 a=inputs,
    #                 perm=[0,2,1],
    #                 conjugate=False,name='SwapFeatureWithHistory'), \
    #           labels
    # else:
    #   return inputs, labels
    return inputs, labels
  
  @tf.autograph.experimental.do_not_convert
  def make_quickData(self, inputs : pd.DataFrame,
                           labels : pd.DataFrame
                ) -> tuple[tf.raw_ops.BatchDataset, tnp.ndarray, tnp.ndarray]:
    tic : float = perf_counter()
    if self.normaliseInput: # Normalise Inputs
      data : pd.DataFrame = (inputs[:-self.input_width]-self.Data.get_Mean[0][self.input_columns])/self.Data.get_STD[0][self.input_columns]
    else:
      data : pd.DataFrame = (inputs[:-self.input_width])
    
    if self.normaliseLabal: # Normalise Labels
      targets : pd.DataFrame = (labels[self.input_width:]-self.Data.get_Mean[1][self.label_columns])/self.Data.get_STD[1][self.label_columns]
    else:
      targets : pd.DataFrame = (labels[self.input_width:])
    
    ds : tf.raw_ops.BatchDataset = \
            tf.keras.preprocessing.timeseries_dataset_from_array(
                  data=data,
                  targets=targets,
                  sequence_length=self.input_width,
                  sequence_stride=1, sampling_rate=1,batch_size=1,
                  shuffle=True, seed=None, start_index=None, end_index=None
            )
    x : tnp.ndarray = tnp.asarray(list(ds.map(
                                lambda x, _: x[0,:,:]
                              ).as_numpy_iterator()
                          ))
    y : tnp.ndarray = tnp.asarray(list(ds.map(
                                lambda _, y: y[0]
                              ).as_numpy_iterator()
                          ))
    print(f"\n\nData windowing took: {(perf_counter() - tic):.2f} seconds")
    return ds, x, y
  
  @tf.autograph.experimental.do_not_convert
  def ParseFullData(self, dir : str
      ) -> tuple[pd.DataFrame, tf.raw_ops.MapDataset, tnp.ndarray, tnp.ndarray]:
    tic : float = perf_counter()
    # Parsing file by file
    if self.includeTarget:
      data_x = tnp.empty(shape=(1,
                            self.input_width,
                            len(self.input_columns)+len(self.label_columns)
                          ),
                      dtype=self.float_dtype
                      )
    else:
      data_x = tnp.empty(shape=(1,
                            self.input_width,
                            len(self.input_columns)
                          ),
                      dtype=self.float_dtype
                      )    
    data_y = tnp.empty(shape=(1,
                          self.input_width,
                          len(self.label_columns)
                        ),
                    dtype=self.float_dtype
                    )
    for _, _, files in os.walk(dir):
      files.sort(key=lambda f: int(f[-13:-5])) # Sort by last dates
      # Initialize empty structures
      data_df : pd.DataFrame = self.Data.Read_Excel_File(dir + '/' + files[0])
      data_SoC: pd.DataFrame = pd.DataFrame(
               data={'SoC' : diffSoC(
                          chargeData=(data_df.loc[:,'Charge_Capacity(Ah)']),
                          discargeData=(data_df.loc[:,'Discharge_Capacity(Ah)'])
                          )},
               dtype=self.float_dtype
            )
      data_SoC['SoC(%)'] = applyMinMax(data_SoC['SoC'])
      #* Converting to Tensor unit
      if self.normaliseInput: # Normalise Inputs
        data : pd.DataFrame = (data_df.copy(deep=True)-self.Data.get_Mean[0][self.input_columns])/self.Data.get_STD[0][self.input_columns]
      else:
        data : pd.DataFrame = (data_df.copy(deep=True))
      
      if self.normaliseLabal: # Normalise Labels
        data[self.label_columns] = (data_SoC[self.label_columns].copy(deep=True)-self.Data.get_Mean[1][self.label_columns])/self.Data.get_STD[1][self.label_columns]
      else:
        data[self.label_columns] = (data_SoC[self.label_columns].copy(deep=True))

      data = data[self.input_columns + self.label_columns] # Ensure order
      data = tnp.array(val=data.values,
              dtype=self.float_dtype, copy=True, ndmin=0)

      data_ds : tf.raw_ops.BatchDataset = \
            tf.keras.preprocessing.timeseries_dataset_from_array(
              data=data, targets=None,
              sequence_length=self.total_window_size, sequence_stride=1,
              sampling_rate=1,
              batch_size=1, shuffle=False,
              seed=None, start_index=None, end_index=None
          )
      
      data_ds : tf.raw_ops.MapDataset = data_ds.map(self.split_window)
      
      data_x : tnp.ndarray = tnp.asarray(list(data_ds.map(
                                  lambda x, _: x[0,:,:]
                                ).as_numpy_iterator()
                            ))
      data_y : tnp.ndarray = tnp.asarray(list(data_ds.map(
                                  lambda _, y: y[0]
                                ).as_numpy_iterator()
                            ))
      if self.batch > 1:
        if self.includeTarget:
          batched_x : tnp.ndarray = tnp.reshape(
                        a=data_x[0:0+self.batch,:,:],
                        newshape=(1,
                                self.batch,
                                len(self.input_columns)+len(self.label_columns)
                                ),
                        order='C'
                      )
        else:
          batched_x : tnp.ndarray = tnp.reshape(
                        a=data_x[0:0+self.batch,:,:],
                        newshape=(1,
                                self.batch,
                                len(self.input_columns)
                                ),
                        order='C'
                      )

        batched_y : tnp.ndarray = tnp.reshape(
                      a=data_y[0:0+self.batch,:,:],
                      newshape=(1,self.batch),
                      order='C'
                    )
        for i in range(1, data_x.shape[0]-self.batch+1):
          if self.includeTarget:
            batched_x = tnp.append(
                            arr=batched_x,
                            values=tnp.reshape(
                                  a=data_x[i:i+self.batch,:,:],
                                  newshape=(1,
                                          self.batch,
                                          len(self.input_columns)+\
                                            len(self.label_columns)
                                          ),
                                  order='C'
                                ),
                            axis=0
                          )
          else:
            batched_x = tnp.append(
                            arr=batched_x,
                            values=tnp.reshape(
                                  a=data_x[i:i+self.batch,:,:],
                                  newshape=(1,
                                          self.batch,
                                          len(self.input_columns)
                                          ),
                                  order='C'
                                ),
                            axis=0)
          batched_y = tnp.append(
                          arr=batched_y,
                          values=tnp.reshape(
                                a=data_y[i:i+self.batch,:,:],
                                newshape=(1,self.batch),
                                order='C'
                              ),
                          axis=0)
        
      for file in files[1:]:
        # Initialize empty structures
        df : pd.DataFrame = self.Data.Read_Excel_File(dir + '/' + file)
        SoC: pd.DataFrame = pd.DataFrame(
                data={'SoC' : diffSoC(
                            chargeData=(df.loc[:,'Charge_Capacity(Ah)']),
                            discargeData=(df.loc[:,'Discharge_Capacity(Ah)'])
                            )},
                dtype=self.float_dtype
              )
        SoC['SoC(%)'] = applyMinMax(SoC['SoC'])
        #* Converting to Tensor unit
        if self.normaliseInput: # Normalise Inputs
          data = (df.copy(deep=True)-self.Data.get_Mean[0][self.input_columns])/self.Data.get_STD[0][self.input_columns]
        else:
          data = (df.copy(deep=True))
        
        if self.normaliseLabal: # Normalise Labels
          data[self.label_columns] = (SoC[self.label_columns].copy(deep=True)-self.Data.get_Mean[1][self.label_columns])/self.Data.get_STD[1][self.label_columns]
        else:
          data[self.label_columns] = (SoC[self.label_columns].copy(deep=True))

        data = data[self.input_columns + self.label_columns] # Ensure order
        data = tnp.array(val=data.values,
                dtype=self.float_dtype, copy=True, ndmin=0)

        ds : tf.raw_ops.BatchDataset = \
              tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data, targets=None,
                sequence_length=self.total_window_size, sequence_stride=1,
                sampling_rate=1,
                batch_size=1, shuffle=False,
                seed=None, start_index=None, end_index=None
            )

        ds : tf.raw_ops.MapDataset = ds.map(self.split_window)
        
        x : tnp.ndarray = tnp.asarray(list(ds.map(
                                    lambda x, _: x[0,:,:]
                                  ).as_numpy_iterator()
                              ))
        y : tnp.ndarray = tnp.asarray(list(ds.map(
                                    lambda _, y: y[0]
                                  ).as_numpy_iterator()
                              ))
              
        data_ds = data_ds.concatenate(dataset=ds)
        data_df = data_df.append(other=df, ignore_index=True)
        data_x = tnp.append(arr=data_x, values=x, axis=0)
        data_y = tnp.append(arr=data_y, values=y, axis=0)
        if self.batch > 1:
          if self.includeTarget:
            bat_x : tnp.ndarray = tnp.reshape(
                                    a=x[0:0+self.batch,:,:],
                                    newshape=(1,
                                            self.batch,
                                            len(self.input_columns)+\
                                              len(self.label_columns)
                                            ),
                                    order='C'
                                  )
          else:
            bat_x : tnp.ndarray = tnp.reshape(
                                    a=x[0:0+self.batch,:,:],
                                    newshape=(1,
                                            self.batch,
                                            len(self.input_columns)
                                            ),
                                    order='C'
                                  )
          bat_y : tnp.ndarray = tnp.reshape(
                                  a=y[0:0+self.batch,:,:],
                                  newshape=(1,self.batch),
                                  order='C'
                                )
          for i in range(1, x.shape[0]-self.batch+1):
            if self.includeTarget:
              bat_x = tnp.append(
                            arr=bat_x,
                            values=tnp.reshape(
                                    a=x[i:i+self.batch,:,:],
                                    newshape=(1,
                                            self.batch,
                                            len(self.input_columns)+\
                                              len(self.label_columns)
                                            ),
                                    order='C'
                                  ),
                            axis=0
                          )
            else:
              bat_x = tnp.append(
                            arr=bat_x,
                            values=tnp.reshape(
                                    a=x[i:i+self.batch,:,:],
                                    newshape=(1,
                                            self.batch,
                                            len(self.input_columns)
                                            ),
                                    order='C'
                                  ),
                            axis=0
                          )
            bat_y = tnp.append(
                          arr=bat_y,
                          values=tnp.reshape(
                                  a=y[i:i+self.batch,:,:],
                                  newshape=(1,self.batch),
                                  order='C'
                                ),
                          axis=0
                        )
          batched_x = tnp.append(arr=batched_x,
                                 values=bat_x,
                                 axis=0)
          batched_y = tnp.append(arr=batched_y,
                                 values=bat_y,
                                 axis=0)

      print(f"\n\nData Generation: {(perf_counter() - tic):.2f} seconds")
      if self.batch > 1:
        print("Returning Batched Datasets")
        return data_df, data_ds, batched_x, batched_y
      else:
        print("Returning Usual Datasets")
        return data_df, data_ds, data_x, data_y
    return None

  @property
  def train(self):
    if (self.shift == 1 & self.label_width == 1 & self.input_width == 1):
      print("Maling train dataset from list")
      x, y = self.make_dataset_from_list(
                                  X=self.Data.train_list,
                                  Y=self.Data.train_list_label
                                )
      return x, y  
    else:
      ds, x, y = self.make_dataset_from_array(
                                  inputs=self.Data.train,
                                  labels=self.Data.train_SoC
                                )
      return ds, x, y

  @property
  def valid(self):
    if (self.shift == 1 & self.label_width == 1 & self.input_width == 1):
      print("Maling train dataset from list")
      x, y = self.make_dataset_from_list(
                                  X=self.Data.valid_list,
                                  Y=self.Data.valid_list_label
                                )
      return x, y
    else:
      ds, x, y = self.make_dataset_from_array(
                                  inputs=self.Data.valid,
                                  labels=self.Data.valid_SoC
                                )
      return ds, x, y  

  @property
  def full_train(self):
    return self.ParseFullData(self.Data.train_dir)
  
  @property
  def full_valid(self):
    return self.ParseFullData(self.Data.valid_dir)