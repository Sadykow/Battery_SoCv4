# %% [markdown]
# # Data Generator
# Implementation of the data extraction based on 3 previos versions.
# Combines Data generator in some implementations.
# %%
import os, time                   # Moving to folders in FS and timing
import pandas as pd               # Tabled data storage
import tensorflow.experimental.numpy as tnp

from dataclasses import dataclass # Used for making annotations
from itertools import chain       # Make Chain ranges
from sklearn.preprocessing import MinMaxScaler
from extractor.soc_calc import diffSoC

c_DST   : range = range(4 ,6 )    # ONLY Charge Cycle
d_DST   : range = range(6 ,12)    # ONLY Desciarge Cycle
r_DST   : range = range(4 ,12)    # Charge-Discharge Continuos cycle

c_US    : range = range(10,14)    # ONLY Charge Cycle
d_US    : range = range(14,20)    # ONLY Desciarge Cycle
r_US    : range = range(10,20)    # Charge-Discharge Continuos cycle

c_FUDS  : range = range(18,22)    # ONLY Charge Cycle
d_FUDS  : range = range(22,25)    # ONLY Desciarge Cycle
r_FUDS  : range = range(18,25)    # Charge-Discharge Continuos cycle


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

  float_dtype   : type          # Float variable Type
  int_dtype     : type          # Int variable Type

  train_df : tnp.ndarray       # Training Dataset   80~85%
  train_t  : float
  train_SoC: tnp.ndarray
  tr_ls_df : list[pd.DataFrame] # List of Training Dataset
  tr_ls_SoC: list[pd.DataFrame]
  train_s  : int
  
  valid_df : tnp.ndarray       # Validating Dataset 15~20%
  valid_t  : float
  valid_SoC: tnp.ndarray
  vl_ls_df : list[pd.DataFrame] # List of Validating Dataset
  vl_ls_SoC: list[pd.DataFrame]
  valid_s  : int
  
  gener_t  : float             # Time it took to create TNP
  #testi_df : pd.DataFrame      # Testing Dataset Any Size%
  
  def __init__(self, train_dir : str, valid_dir : str, test_dir : str,
               columns : list[str],
               PROFILE_range : str,
               float_dtype : type = tnp.float32,
               int_dtype : type = tnp.int16) -> None:
    """ Data Constructor used to extract Excel files by profiles

    Args:
        train_dir (str): [description]
        valid_dir (str): [description]
        test_dir (str): [description]
        columns (list[str]): [description]
        PROFILE_range (str): [description]
        float_dtype (type, optional): [description]. Defaults to tnp.float32.
        int_dtype (type, optional): [description]. Defaults to tnp.int16.
    """
    # Store the raw data information
    self.train_dir = train_dir
    self.valid_dir = valid_dir
    self.testi_dir = test_dir
    self.columns = columns
    
    #! Charge and Discharge only into this switch
    # Select profile based on string
    if(PROFILE_range == 'DST'):
      self.r_profile = r_DST
      self.v_profile = r_US_FUDS
    elif(PROFILE_range == 'US06'):
      self.r_profile = r_US
      self.v_profile = None
    elif(PROFILE_range == 'FUDS'):
      self.r_profile = r_FUDS
      self.v_profile = r_DST_US
    elif(PROFILE_range == 'd_DST'):
      self.r_profile = d_DST
      self.v_profile = r_US_FUDS
    elif(PROFILE_range == 'd_US06'):
      self.r_profile = d_US
      self.v_profile = None
    elif(PROFILE_range == 'd_FUDS'):
      self.r_profile = d_FUDS
      self.v_profile = r_DST_US
    else:
      self.r_profile = r_DST_US_FUDS
      self.v_profile = r_DST_US
    
    # Variable types to use
    self.float_dtype = float_dtype
    self.int_dtype = int_dtype

    # Extracting data
    tic : float = time.perf_counter()
    self.tr_ls_df, self.tr_ls_SoC = self.ParseExcelData(self.train_dir,
                                                        self.r_profile)
    self.train_t = time.perf_counter() - tic
    
    tic : float = time.perf_counter()
    self.vl_ls_df, self.vl_ls_SoC = self.ParseExcelData(self.valid_dir,
                                                        self.v_profile)
    self.valid_t = time.perf_counter() - tic
    
    # Get number of samples
    self.train_s = 0
    self.valid_s = 0
    for i in range(0, len(self.tr_ls_df)):
      self.train_s += self.tr_ls_df[i].shape[0]    
    for i in range(0, len(self.vl_ls_df)):
      self.valid_s += self.vl_ls_df[i].shape[0]
    
    # Creating TensorNumpy arrays for all dataset
    scaller : MinMaxScaler = MinMaxScaler(feature_range=(0,1))
    tic : float = time.perf_counter()
    self.train_df = tnp.array(val=self.tr_ls_df[0],
                              dtype=self.float_dtype, copy=True)
    self.train_SoC = tnp.array(val=scaller.fit_transform(self.tr_ls_SoC[0]),
                              dtype=self.float_dtype, copy=True)
    self.valid_df = tnp.array(val=self.vl_ls_df[0],
                              dtype=self.float_dtype, copy=True)
    self.valid_SoC = tnp.array(val=scaller.fit_transform(self.vl_ls_SoC[0]),
                              dtype=self.float_dtype, copy=True)
    for i in range(1, len(self.tr_ls_df)):
      self.train_df = tnp.append(
                              arr=self.train_df,
                              values=tnp.array(
                                    val=self.tr_ls_df[i],
                                    dtype=self.float_dtype, copy=True
                                  ),
                              axis=0
                            )
      self.train_SoC = tnp.append(
                              arr=self.train_SoC,
                              values=tnp.array(
                                  val=scaller.fit_transform(self.tr_ls_SoC[i]),
                                  dtype=self.float_dtype, copy=True
                                ),
                              axis=0
                            )
    for i in range(1, len(self.vl_ls_df)):
      self.valid_df = tnp.append(
                              arr=self.valid_df,
                              values=tnp.array(
                                    val=self.vl_ls_df[i],
                                    dtype=self.float_dtype, copy=True
                                  ),
                              axis=0
                            )
      self.valid_SoC = tnp.append(
                              arr=self.valid_SoC,
                              values=tnp.array(
                                  val=scaller.fit_transform(self.vl_ls_SoC[i]),
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
    total_samples = self.train_s + self.valid_s
    train_proportion = self.train_s/total_samples*100
    valid_proportion = self.valid_s/total_samples*100
    return '\n'.join([
      f'\n\n№ Trainig Samples PD, NP: {self.train_s}, {self.train_df.shape[0]}',
      f'Time took to extract Tr: {self.train_t}',
      f'№ Validation Samples  PD, NP: {self.valid_s}, {self.valid_df.shape[0]}',
      f'Time took to extract Vl: {self.valid_t}',
      f'Data Vl/Tr: {valid_proportion:.2f}% to {train_proportion:.2f}%',
      f'Time for TNP generator: {self.gener_t}',
      f'Data Mean: {tnp.mean(self.train_df)}',
      f'Data STD: {tnp.std(self.train_df)}',
      '\n\n'])

  def ParseExcelData(self, directory : str,
                    indexes : range,
              ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """ Parsing Excel data from Battery Testing Machine

    Args:
        directory (str): Dataset directory location. !! Make sure not other file
    formats stored. No check has been added.
        indexes (range): The data indexes to select regions

    Returns:
        tuple[list[pd.DataFrame], list[pd.DataFrame]]: Returning data itself 
    and SoC in the list format to capture absolutely all samples.
    """
    for _, _, files in os.walk(directory):
      files.sort(key=lambda f: int(f[-13:-5])) # Sort by last dates
      # Initialize empty structures
      data_df : list[pd.DataFrame] = []
      data_SoC : list[pd.DataFrame] = []
      # data_df : pd.DataFrame = self.Read_Excel_File(directory + '/' + files[0],
      #                                               indexes)
      # data_SoC: pd.DataFrame = pd.DataFrame(
      #          data={'SoC' : diffSoC(
      #                     chargeData=(data_df.loc[:,'Charge_Capacity(Ah)']),
      #                     discargeData=(data_df.loc[:,'Discharge_Capacity(Ah)'])
      #                     )},
      #          dtype=self.float_dtype
      #       )
      # data_SoC['SoC(%)'] = MinMaxScaler(data_SoC['SoC'])

      for file in files[:]:
        df : pd.DataFrame = self.Read_Excel_File(directory + '/' + file,
                                                    indexes)
        SoC: pd.DataFrame = pd.DataFrame(
               data={'SoC' : diffSoC(
                            chargeData=df.loc[:,'Charge_Capacity(Ah)'],
                            discargeData=df.loc[:,'Discharge_Capacity(Ah)']
                            )},
               dtype=self.float_dtype
            )
        #SoC['SoC(%)'] = MinMaxScaler(SoC['SoC'])

        #data_df = data_df.append(df.copy(deep=True), ignore_index=True)
        #data_SoC = data_SoC.append(SoC.copy(deep=True), ignore_index=True)
        data_df.append(df)
        data_SoC.append(SoC)
      return data_df, data_SoC

  def Read_Excel_File(self, path : str, indexes : range) -> pd.DataFrame:
    """ Reads Excel File with all parameters. Sheet Name universal, columns,
    type taken from global variables initialization.

    Args:
        path (str): Path to files with os.walk
        indexes (range): Step_Index to select from

    Returns:
        pd.DataFrame: Single File frame.
    """
    if(indexes == None):       #!r_DST_FUDS Nasty fix
      indexes = chain(range(4 , 12), range(18, 25))
    try:
      df : pd.DataFrame = pd.read_excel(io=path,
                        sheet_name='Channel_1-006',
                        header=0, names=None, index_col=None,
                        usecols=['Step_Index'] + self.columns,
                        squeeze=False,
                        dtype=self.float_dtype,
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
                        usecols=['Step_Index'] + self.columns,
                        squeeze=False,
                        dtype=self.float_dtype,
                        engine='openpyxl', converters=None, true_values=None,
                        false_values=None, skiprows=None, nrows=None,
                        na_values=None, keep_default_na=True, na_filter=True,
                        verbose=False, parse_dates=False, date_parser=None,
                        thousands=None, comment=None, skipfooter=0,
                        convert_float=True, mangle_dupe_cols=True
                      )
    df = df[df['Step_Index'].isin(indexes)]
    df = df.reset_index(drop=True)
    df = df.drop(columns=['Step_Index'])
    df = df[self.columns]   # Order columns in the proper sequence
    return df

  @property
  def train_list(self) -> tnp.ndarray:
    return self.tr_ls_df

  @property
  def train(self) -> tnp.ndarray:
    """ Training Dataset

    Returns:
        pd.DataFrame: train['Current', 'Voltage' ...]
    """
    return self.train_df

  @property
  def train_label(self) -> tnp.ndarray:
    """ Training Dataset

    Returns:
        pd.DataFrame: train_SoC['SoC', 'SoC(%)' ...]
    """
    return self.train_SoC

  @property
  def train_list_label(self) -> tnp.ndarray:
    return self.tr_ls_SoC

  @property
  def valid(self) -> tnp.ndarray:
    """ Validation Dataset

    Returns:
        pd.DataFrame: valid['Current', 'Voltage' ...]
    """
    return self.valid_df

  @property
  def valid_list(self) -> tnp.ndarray:
    return self.vl_ls_df

  @property
  def valid_label(self) -> tnp.ndarray:
    """ Validation Dataset

    Returns:
        pd.DataFrame: valid['SoC', 'SoC(%)' ...]
    """
    return self.valid_SoC
  
  @property
  def valid_list_label(self) -> tnp.ndarray:
    """ Validation Dataset

    Returns:
        pd.DataFrame: valid['SoC', 'SoC(%)' ...]
    """
    return self.vl_ls_SoC