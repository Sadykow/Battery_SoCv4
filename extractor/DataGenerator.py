# %% [markdown]
# # Data Generator
# Implementation of the data extraction based on 3 previos versions.
# Combines Data generator in some implementations.
# %%
import os, sys, time              # Moving to folders in FS and timing
import pandas as pd               # Tabled data storage
import numpy as np

from dataclasses import dataclass # Used for making annotations
from itertools import chain       # Make Chain ranges
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.getcwd() + '/..')
from py_modules.parse_excel import ParseExcelData

if (sys.version_info[1] < 9):
  from typing import List as list
  from typing import Tuple as tuple
  
c_DST   : range = [3, 4, 5, 7]    # ONLY Charge Cycle
d_DST   : range = range(8 ,10)    # ONLY Desciarge Cycle
r_DST   : range = [range(3 ,8),
                   range(8 ,12)]  # Charge-Discharge Continuos cycle

c_US    : range = [11, 12, 13, 15]# ONLY Charge Cycle
d_US    : range = range(16,18)    # ONLY Desciarge Cycle
r_US    : range = [range(11,16),  
                   range(16,20)]  # Charge-Discharge Continuos cycle

c_FUDS  : range = [19, 20, 21, 23]# ONLY Charge Cycle
d_FUDS  : range = range(24,26)    # ONLY Desciarge Cycle
r_FUDS  : range = [range(19,24),
                   range(24,28)]  # Charge-Discharge Continuos cycle


r_US_FUDS     : range = range(10, 25)  # US06 and FUDS
#!r_DST_FUDS bug with chain() nasty used as hot fix
r_DST_US      : range = range(4 , 20)  # DST and US06
r_DST_US_FUDS : range = range(4 , 25)  # Full cycle
@dataclass
class DataGenerator():
  train_dir     : str
  valid_dir     : str           # Data directory location
  testi_dir     : str
  columns       : list[str]     # Columns names and their order
  r_profile     : range         # Profile to use for Training
  v_profile     : range         # Profile to use for Validation
  t_profile     : range         # Profile to use for Testing

  float_dtype   : type          # Float variable Type
  int_dtype     : type          # Int variable Type

  train_df : np.ndarray       # Training Dataset   80~85%
  train_t  : float
  train_SoC: np.ndarray
  tr_ls_df : list[pd.DataFrame] # List of Training Dataset
  tr_ls_SoC: list[pd.DataFrame]
  train_s  : int
  
  valid_df : np.ndarray       # Validating Dataset 15~20%
  valid_t  : float
  valid_SoC: np.ndarray
  vl_ls_df : list[pd.DataFrame] # List of Validating Dataset
  vl_ls_SoC: list[pd.DataFrame]
  valid_s  : int

  testi_df : np.ndarray       # Testing Dataset Oposite profiles
  testi_t  : float
  testi_SoC: np.ndarray
  ts_ls_df : list[pd.DataFrame] # List of testing Dataset
  ts_ls_SoC: list[pd.DataFrame]
  testi_s  : int

  gener_t  : float             # Time it took to create np
  
  spacing  : int = 5           # Sampeling at charge to resample

  def __init__(self, train_dir : str, valid_dir : str, test_dir : str,
               columns : list[str],
               PROFILE_range : str,
               round   : int = 4,
               float_dtype : type = np.float32,
               int_dtype : type = np.int16) -> None:
    """ Data Constructor used to extract Excel files by profiles

    Args:
        train_dir (str): [description]
        valid_dir (str): [description]
        test_dir (str): [description]
        columns (list[str]): [description]
        PROFILE_range (str): [description]
        round (int): [description]
        float_dtype (type, optional): [description]. Defaults to np.float32.
        int_dtype (type, optional): [description]. Defaults to np.int16.
    """
    # Store the raw data information
    self.train_dir = train_dir
    self.valid_dir = valid_dir
    self.testi_dir = test_dir
    self.columns = columns
    
    #! Charge and Discharge only into this switch
    # Select profile based on string
    if(PROFILE_range == 'DST'):
      self.r_profile = self.v_profile = [c_DST, d_DST]
      self.t_profile = [r_US, r_FUDS] #r_US_FUDS #! Test contains outliers
    elif(PROFILE_range == 'US06'):
      self.r_profile = r_US
      self.v_profile = r_US
      self.t_profile = [r_DST, r_FUDS]     #! A stub to resolve DST and FUDS
    elif(PROFILE_range == 'FUDS'):
      self.r_profile = r_FUDS
      self.v_profile = r_FUDS
      self.t_profile = [r_DST, r_US]#r_DST_US
    elif(PROFILE_range == 'd_DST'):
      self.r_profile = d_DST
      self.v_profile = d_DST
      self.t_profile = [r_US, r_FUDS]#r_US_FUDS
    elif(PROFILE_range == 'd_US06'):
      self.r_profile = d_US
      self.v_profile = d_US
      self.t_profile = None
    elif(PROFILE_range == 'd_FUDS'):
      self.r_profile = d_FUDS
      self.v_profile = d_FUDS
      self.t_profile = r_DST_US
    else:
      self.r_profile = r_DST_US_FUDS
      self.v_profile = r_DST_US_FUDS
      self.t_profile = r_DST_US
    
    # Variable types to use
    self.float_dtype = float_dtype
    self.int_dtype = int_dtype

    # Extracting data
    #* Training
    tic : float = time.perf_counter()
    self.tr_ls_df, self.tr_ls_SoC = ParseExcelData(self.train_dir,
                                                   self.r_profile[0],
                                                   self.columns)
    self.interpolate_charge(self.tr_ls_df, self.spacing)
    self.interpolate_charge(self.tr_ls_SoC, self.spacing, noise=False)
    ds_ls_df, ds_ls_SoC = ParseExcelData(self.train_dir,
                                         self.r_profile[1],
                                         self.columns)
    for i in range(len(self.tr_ls_df)):
      self.tr_ls_df[i] = self.tr_ls_df[i].append(ds_ls_df[i])
      self.tr_ls_df[i].reset_index(drop=True, inplace=True)
      self.tr_ls_SoC[i] = self.tr_ls_SoC[i].append(ds_ls_SoC[i])
      self.tr_ls_SoC[i].reset_index(drop=True, inplace=True)
    self.train_t = time.perf_counter() - tic
    
    #* Validation
    tic : float = time.perf_counter()
    self.vl_ls_df, self.vl_ls_SoC = ParseExcelData(self.valid_dir,
                                                   self.v_profile[0],
                                                   self.columns)
    self.interpolate_charge(self.vl_ls_df, self.spacing)
    self.interpolate_charge(self.vl_ls_SoC, self.spacing, noise=False)
    ds_ls_df, ds_ls_SoC = ParseExcelData(self.valid_dir,
                                         self.v_profile[1],
                                         self.columns)
    for i in range(len(self.vl_ls_df)):
      self.vl_ls_df[i] = self.vl_ls_df[i].append(ds_ls_df[i])
      self.vl_ls_df[i].reset_index(drop=True, inplace=True)
      self.vl_ls_SoC[i] = self.vl_ls_SoC[i].append(ds_ls_SoC[i])
      self.vl_ls_SoC[i].reset_index(drop=True, inplace=True)
    self.valid_t = time.perf_counter() - tic
    
    #* Testing
    #** Part 1
    tic : float = time.perf_counter()
    self.ts_ls_df, self.ts_ls_SoC = ParseExcelData(self.testi_dir,
                                                   self.t_profile[0][0],
                                                   self.columns)
    self.interpolate_charge(self.ts_ls_df, self.spacing)
    self.interpolate_charge(self.ts_ls_SoC, self.spacing, noise=False)
    ds_ls_df, ds_ls_SoC = ParseExcelData(self.testi_dir,
                                         self.t_profile[0][1],
                                         self.columns)
    for i in range(len(self.ts_ls_df)):
      self.ts_ls_df[i] = self.ts_ls_df[i].append(ds_ls_df[i])
      self.ts_ls_df[i].reset_index(drop=True, inplace=True)
      self.ts_ls_SoC[i] = self.ts_ls_SoC[i].append(ds_ls_SoC[i])
      self.ts_ls_SoC[i].reset_index(drop=True, inplace=True)
    
    #** Part 2
    ts_ls_df_2, ts_ls_SoC_2 = ParseExcelData(self.testi_dir,
                                             self.t_profile[1][0],
                                             self.columns)
    self.interpolate_charge(ts_ls_df_2, self.spacing)
    self.interpolate_charge(ts_ls_SoC_2, self.spacing, noise=False)
    ds_ls_df, ds_ls_SoC = ParseExcelData(self.testi_dir,
                                         self.t_profile[1][1],
                                         self.columns)
    for i in range(len(ts_ls_df_2)):
      ts_ls_df_2[i] = ts_ls_df_2[i].append(ds_ls_df[i])
      ts_ls_df_2[i].reset_index(drop=True, inplace=True)
      ts_ls_SoC_2[i] = ts_ls_SoC_2[i].append(ds_ls_SoC[i])
      ts_ls_SoC_2[i].reset_index(drop=True, inplace=True)

    #** Combine two lists to one
    self.ts_ls_df.extend(ts_ls_df_2)
    self.ts_ls_SoC.extend(ts_ls_SoC_2)
    self.testi_t = time.perf_counter() - tic

    # Get number of samples and rounding columns
    self.train_s = 0
    self.valid_s = 0
    self.testi_s = 0
    for i in range(0, len(self.tr_ls_df)):
      self.tr_ls_df[i] = self.tr_ls_df[i].round(decimals=round)
      self.train_s += self.tr_ls_df[i].shape[0]    
    for i in range(0, len(self.vl_ls_df)):
      self.vl_ls_df[i] = self.vl_ls_df[i].round(decimals=round)
      self.valid_s += self.vl_ls_df[i].shape[0]
    for i in range(0, len(self.ts_ls_df)):
      self.ts_ls_df[i] = self.ts_ls_df[i].round(decimals=round)
      self.testi_s += self.ts_ls_df[i].shape[0]

    # Creating Numpy arrays for all dataset
    #! Ensure that MinMax works with below zero values.
    scaller : MinMaxScaler = MinMaxScaler(feature_range=(0,1))
    tic : float = time.perf_counter()
    #* Training
    self.train_df = np.array(object=self.tr_ls_df[0],
                              dtype=self.float_dtype, copy=True)
    self.tr_ls_SoC[0] = scaller.fit_transform(self.tr_ls_SoC[0])
    self.tr_ls_SoC[0] = self.tr_ls_SoC[0].round(decimals=round)
    self.train_SoC = np.array(object=self.tr_ls_SoC[0],
                              dtype=self.float_dtype, copy=True)
    #* Valid
    self.valid_df = np.array(object=self.vl_ls_df[0],
                              dtype=self.float_dtype, copy=True)
    self.vl_ls_SoC[0] = scaller.fit_transform(self.vl_ls_SoC[0])
    self.vl_ls_SoC[0] = self.vl_ls_SoC[0].round(decimals=round)
    self.valid_SoC = np.array(object=self.vl_ls_SoC[0],
                              dtype=self.float_dtype, copy=True)
    #* Test
    self.testi_df = np.array(object=self.ts_ls_df[0],
                              dtype=self.float_dtype, copy=True)
    self.ts_ls_SoC[0] = scaller.fit_transform(self.ts_ls_SoC[0])
    self.ts_ls_SoC[0] = self.ts_ls_SoC[0].round(decimals=round)
    self.testi_SoC = np.array(object=self.ts_ls_SoC[0],
                              dtype=self.float_dtype, copy=True)

    for i in range(1, len(self.tr_ls_df)):
      self.train_df = np.append(
                              arr=self.train_df,
                              values=np.array(
                                    object=self.tr_ls_df[i],
                                    dtype=self.float_dtype, copy=True
                                  ),
                              axis=0
                            )
      self.tr_ls_SoC[i] = scaller.fit_transform(self.tr_ls_SoC[i])
      self.tr_ls_SoC[i] = self.tr_ls_SoC[i].round(decimals=round)
      self.train_SoC = np.append(
                              arr=self.train_SoC,
                              values=np.array(
                                  object=self.tr_ls_SoC[i],
                                  dtype=self.float_dtype, copy=True
                                ),
                              axis=0
                            )
    for i in range(1, len(self.vl_ls_df)):
      self.valid_df = np.append(
                              arr=self.valid_df,
                              values=np.array(
                                    object=self.vl_ls_df[i],
                                    dtype=self.float_dtype, copy=True
                                  ),
                              axis=0
                            )
      self.vl_ls_SoC[i] = scaller.fit_transform(self.vl_ls_SoC[i])
      self.vl_ls_SoC[i] = self.vl_ls_SoC[i].round(decimals=round)
      self.valid_SoC = np.append(
                              arr=self.valid_SoC,
                              values=np.array(
                                  object=self.vl_ls_SoC[i],
                                  dtype=self.float_dtype, copy=True
                                ),
                              axis=0
                            )
    for i in range(1, len(self.ts_ls_df)):
      self.testi_df = np.append(
                              arr=self.testi_df,
                              values=np.array(
                                    object=self.ts_ls_df[i],
                                    dtype=self.float_dtype, copy=True
                                  ),
                              axis=0
                            )
      self.ts_ls_SoC[i] = scaller.fit_transform(self.ts_ls_SoC[i])
      self.ts_ls_SoC[i] = self.ts_ls_SoC[i].round(decimals=round)
      self.testi_SoC = np.append(
                              arr=self.testi_SoC,
                              values=np.array(
                                  object=self.ts_ls_SoC[i],
                                  dtype=self.float_dtype, copy=True
                                ),
                              axis=0
                            )
    self.gener_t = time.perf_counter() - tic    

  def __repr__(self) -> str:
    """ General information upon how many samples per each dataset and time in
    took to create one. Along with proportion.

    Returns:
        str: New line string with information.
    """
    total_samples = self.train_s + self.valid_s + self.testi_s
    train_proportion = self.train_s/total_samples*100
    valid_proportion = self.valid_s/total_samples*100
    return '\n'.join([
      f'\n\n№ Trainig Samples PD, NP: {self.train_s}, {self.train_df.shape[0]}',
      f'Time took to extract Tr: {self.train_t}',
      f'№ Validation Samples  PD, NP: {self.valid_s}, {self.valid_df.shape[0]}',
      f'Time took to extract Vl: {self.valid_t}',
      f'Data Vl/Tr: {valid_proportion:.2f}% to {train_proportion:.2f}%',
      f'Time for np generator: {self.gener_t}',
      f'Data Mean: {np.mean(self.train_df)}',
      f'Data STD: {np.std(self.train_df)}',
      '\n\n'])

  def interpolate_data(self, ls_df : list[np.ndarray], spacing : int):
    #TODO: Use separate interpolation for V and T
    #TODO: method='polynomial', order=5      
    for i in range(len(ls_df)):
      ls_df[i]['reindex'] = np.arange(0, spacing*len(ls_df[i]), spacing)
      ls_df[i] = ls_df[i].set_index('reindex').reindex(
          np.arange(0, spacing*len(ls_df[i]), 1)
        )
      ls_df[i]['Current(A)'] = ls_df[i]['Current(A)'].interpolate('linear')
      ls_df[i]['Voltage(V)'] = ls_df[i]['Voltage(V)'].interpolate(
                                  method='polynomial', order=15
                                )
      ls_df[i]['Temperature (C)_1'] = ls_df[i]['Temperature (C)_1'].interpolate(
                                  method='polynomial', order=5
                                )
      ls_df[i].iloc[:, -2:] = ls_df[i].iloc[:, -2:].interpolate('linear')
      ls_df[i].reset_index(drop=True, inplace=True)

  def interpolate_charge(self, ls_df : list[np.ndarray], spacing : int,
                         noise : bool = True):
    # 
    # a = -diff
    # b = diff
    # temp = round( a + (b-a) * np.random.sample(7), 2)
    # if any(temp.round(2) > -0.01) | any(temp.round(2) < 0.01):
    #   print('bad')
    for i in range(len(ls_df)):
      # Prep noise
      if noise:
        diff = (ls_df[i].copy()-ls_df[i].shift().copy())[1:].copy()
        diff = pd.concat([diff, diff.iloc[-1:, :]])
        # diff=diff*0.25 # 25% <<<33.4% offset
        spacing = 5
        diff['reindex'] = np.arange(0, spacing*len(diff), spacing)
        diff = diff.set_index('reindex').reindex(
            np.arange(0, spacing*len(diff), 1)
          ).interpolate('pad')
        diff.reset_index(drop=True, inplace=True)
        
        sign = np.zeros(shape=diff.shape)
        b = 0.25
        a = -b
        for j in range(sign.shape[0]):
          if j % spacing == 0:
            s = np.zeros(shape=sign.shape[1])
          else:
            # s = np.random.randint(low=-1, high=2,
            #                     size=sign.shape[1])
            s = np.random.uniform(a, b, sign.shape[1]).round(4)
            # while( (s == 0).any() ):
            while( any(s > -0.0001) & any(s < 0.0001) ):
              # s = np.random.randint(low=-1, high=2,
              #                       size=sign.shape[1])
              s = np.random.uniform(a, b, sign.shape[1]).round(4)
          sign[j, :] = s
        diff = diff*sign
      ls_df[i]['reindex'] = np.arange(0, spacing*len(ls_df[i]), spacing)
      ls_df[i] = ls_df[i].set_index('reindex').reindex(
          np.arange(0, spacing*len(ls_df[i]), 1)
        ).interpolate('linear')
      ls_df[i].reset_index(drop=True, inplace=True)
      
      if noise:
        ls_df[i] = ls_df[i].add(diff)

  #* Training getters
  @property
  def train_list(self) -> np.ndarray:
    return self.tr_ls_df

  @property
  def train(self) -> np.ndarray:
    """ Training Dataset

    Returns:
        pd.DataFrame: train['Current', 'Voltage' ...]
    """
    return self.train_df

  @property
  def train_label(self) -> np.ndarray:
    """ Training Dataset

    Returns:
        pd.DataFrame: train_SoC['SoC', 'SoC(%)' ...]
    """
    return self.train_SoC

  @property
  def train_list_label(self) -> np.ndarray:
    return self.tr_ls_SoC

  #* Validation getters
  @property
  def valid(self) -> np.ndarray:
    """ Validation Dataset

    Returns:
        pd.DataFrame: valid['Current', 'Voltage' ...]
    """
    return self.valid_df

  @property
  def valid_list(self) -> np.ndarray:
    return self.vl_ls_df

  @property
  def valid_label(self) -> np.ndarray:
    """ Validation Dataset

    Returns:
        pd.DataFrame: valid['SoC', 'SoC(%)' ...]
    """
    return self.valid_SoC
  
  @property
  def valid_list_label(self) -> np.ndarray:
    """ Validation Dataset

    Returns:
        pd.DataFrame: valid['SoC', 'SoC(%)' ...]
    """
    return self.vl_ls_SoC

  #* Testing getters
  @property
  def testi(self) -> np.ndarray:
    """ Testing Dataset

    Returns:
        pd.DataFrame: testi['Current', 'Voltage' ...]
    """
    return self.testi_df

  @property
  def testi_list(self) -> np.ndarray:
    return self.ts_ls_df

  @property
  def testi_label(self) -> np.ndarray:
    """ Testing Dataset

    Returns:
        pd.DataFrame: testi['SoC', 'SoC(%)' ...]
    """
    return self.testi_SoC
  
  @property
  def testi_list_label(self) -> np.ndarray:
    """ Testing Dataset

    Returns:
        pd.DataFrame: testi['SoC', 'SoC(%)' ...]
    """
    return self.ts_ls_SoC