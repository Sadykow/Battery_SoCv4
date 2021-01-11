# %% [markdown]
# # Data Generator
# Implementation of the data extraction based on 3 previos versions.
# Combines Data generator in some implementations.
# %%
import os, time                   # Moving to folders in FS and timing
import pandas as pd               # Tabled data storage
import tensorflow.experimental.numpy as tnp

from dataclasses import dataclass # Used for making annotations

from parser.soc_calc import *

#r_FUDS  : range = range(18,25)
r_FUDS  : range = range(21,25)
r_DST_US_FUDS : range = range(5, 25)

@dataclass
class DataGenerator():
  train_dir     : str
  valid_dir     : str           # Data directory location
  testi_dir     : str
  columns       : list[str]     # Columns names and their order
  r_profile     : str           # Profile to use as String

  float_dtype   : type          # Float variable Type
  int_dtype     : type          # Int variable Type

  train_df : pd.DataFrame      # Training Dataset   80~85%
  train_t  : float
  train_SoC: pd.DataFrame
  
  valid_df : pd.DataFrame      # Validating Dataset 15~20%
  valid_t  : float
  valid_SoC: pd.DataFrame
  
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
    
    # Select profile based on string
    if(PROFILE_range == 'FUDS'):
      self.r_profile = r_FUDS
    elif(PROFILE_range == 'DST_US06_FUDS'):
      self.r_profile == r_DST_US_FUDS
    else:
      self.r_profile == r_FUDS
    
    # Variable types to use
    self.float_dtype = float_dtype
    self.int_dtype = int_dtype

    # Extracting data
    tic : float = time.perf_counter()
    self.train_df, self.train_SoC = self.ParseExcelData(self.train_dir)
    self.train_t = time.perf_counter() - tic
    
    tic : float = time.perf_counter()
    self.valid_df, self.valid_SoC = self.ParseExcelData(self.valid_dir)
    self.valid_t = time.perf_counter() - tic
  
  def __repr__(self) -> str:
    """ General information upon how many samples per each dataset and time in
    took to create one. Along with proportion.

    Returns:
        str: New line string with information.
    """
    total_samples = self.train_df.shape[0] + self.valid_df.shape[0]
    train_proportion = self.train_df.shape[0]/total_samples*100
    valid_proportion = self.valid_df.shape[0]/total_samples*100
    return '\n'.join([
      f'\n\n№ Trainig Samples: {self.train_df.shape[0]}',
      f'Time took to extract Tr: {self.train_t}',
      f'№ Validation Samples: {self.valid_df.shape[0]}',
      f'Time took to extract Vl: {self.valid_t}',
      f'Data Vl/Tr: {valid_proportion:.2f}% to {train_proportion:.2f}%',
      '\n\n'])

  def ParseExcelData(self, directory : str
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Parsing Excel data from Battery Testing Machine

    Args:
        directory (str): Dataset directory location. !! Make sure not other file
    formats stored. No check has been added.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Returning data itself and SoC
    """
    for _, _, files in os.walk(directory):
      files.sort(key=lambda f: int(f[-13:-5])) # Sort by last dates
      # Initialize empty structures
      data_df : pd.DataFrame = self.Read_Excel_File(directory + '/' + files[0])
      data_SoC: pd.DataFrame = pd.DataFrame(
               data={'SoC' : diffSoC(
                          chargeData=(data_df.loc[:,'Charge_Capacity(Ah)']),
                          discargeData=(data_df.loc[:,'Discharge_Capacity(Ah)'])
                          )},
               dtype=self.float_dtype
            )
      data_SoC['SoC(%)'] = applyMinMax(data_SoC['SoC'])

      for file in files[1:]:
        df : pd.DataFrame = self.Read_Excel_File(directory + '/' + file)
        SoC: pd.DataFrame = pd.DataFrame(
               data={'SoC' : diffSoC(
                            chargeData=df.loc[:,'Charge_Capacity(Ah)'],
                            discargeData=df.loc[:,'Discharge_Capacity(Ah)']
                            )},
               dtype=self.float_dtype
            )
        SoC['SoC(%)'] = applyMinMax(SoC['SoC'])

        data_df = data_df.append(df.copy(deep=True), ignore_index=True)
        data_SoC = data_SoC.append(SoC.copy(deep=True), ignore_index=True)
      return data_df, data_SoC

  def Read_Excel_File(self, path : str) -> pd.DataFrame:
    """ Reads Excel File with all parameters. Sheet Name universal, columns,
    type taken from global variables initialization.

    Args:
        path (str): Path to files with os.walk

    Returns:
        pd.DataFrame: Single File frame.
    """
    try:
      df : pd.DataFrame = pd.read_excel(io=path,
                        sheet_name='Channel_1-006',
                        header=0, names=None, index_col=None,
                        usecols=['Step_Index'] + self.columns,
                        squeeze=False,
                        dtype=self.float_dtype,
                        engine=None, converters=None, true_values=None,
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
                        engine=None, converters=None, true_values=None,
                        false_values=None, skiprows=None, nrows=None,
                        na_values=None, keep_default_na=True, na_filter=True,
                        verbose=False, parse_dates=False, date_parser=None,
                        thousands=None, comment=None, skipfooter=0,
                        convert_float=True, mangle_dupe_cols=True
                      )
    df = df[df['Step_Index'].isin(self.r_profile)]
    df = df.reset_index(drop=True)
    df = df.drop(columns=['Step_Index'])
    df = df[self.columns]   # Order columns in the proper sequence
    return df

  @property
  def get_Mean(self) -> tuple[pd.Series, pd.Series]:
    """ Gets the mean of training data. Normalization has to be performed only
    by training constants.

    Returns:
        tuple[pd.Series, pd.Series]: Separate mean for Data and SoC
    """
    return self.train_df.mean(), self.train_SoC.mean()
  
  @property
  def get_STD(self) -> tuple[pd.Series, pd.Series]:
    """Get the Standard Deviation of training data. Normalization has to be
    performed only on training constants.

    Returns:
        tuple[pd.Series, pd.Series]: Separate STD for DATA and SoC
    """
    return self.train_df.std(), self.train_SoC.std()
  
  @property
  def train(self) -> pd.DataFrame:
    """ Training Dataset

    Returns:
        pd.DataFrame: train['Current', 'Voltage' ...]
    """
    return self.train_df

  @property
  def train_label(self) -> pd.DataFrame:
    """ Training Dataset

    Returns:
        pd.DataFrame: train_SoC['SoC', 'SoC(%)' ...]
    """
    return self.train_SoC

  @property
  def valid(self) -> pd.DataFrame:
    """ Validation Dataset

    Returns:
        pd.DataFrame: valid['Current', 'Voltage' ...]
    """
    return self.valid_df

  @property
  def valid_label(self) -> pd.DataFrame:
    """ Validation Dataset

    Returns:
        pd.DataFrame: valid['SoC', 'SoC(%)' ...]
    """
    return self.valid_SoC