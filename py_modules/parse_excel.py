import os                         # Moving to folders in FS and timing
import pandas as pd               # Tabled data storage
from numpy import float32
from itertools import chain, repeat       # Make Chain ranges

# Threadpool
import concurrent.futures

from . utils import diffSoC

def Read_Excel_File(path : str,
                    indexes : range, columns :list[str]
                    ) -> pd.DataFrame:
  """ Reads Excel File with all parameters. Sheet Name universal, columns,
    type taken from global variables initialization.

  Args:
      path (str): Path to files with os.walk
      indexes (range): Step_Index to select from
      columns (list[str]): List of columns in String format

  Returns:
      pd.DataFrame: Single file out
  """
  df : pd.DataFrame = pd.read_excel(io=path,
                      sheet_name=1,
                      header=0, names=None, index_col=None,
                      usecols=['Step_Index'] + columns,
                      squeeze=False,
                      dtype=float32,
                      engine='openpyxl', converters=None, true_values=None,
                      false_values=None, skiprows=None, nrows=None,
                      na_values=None, keep_default_na=True, na_filter=True,
                      verbose=False, parse_dates=False, date_parser=None,
                      thousands=None, comment=None, skipfooter=0,
                      convert_float=True, mangle_dupe_cols=True
                  )
  if(indexes == None):       #!r_DST_FUDS Nasty fix
    df = df[df['Step_Index'].isin(chain(range(4 , 12), range(18, 25)))]
  else:
    df = df[df['Step_Index'].isin(indexes)]
  df = df.reset_index(drop=True)
  df = df.drop(columns=['Step_Index'])
  df = df[columns]   # Order columns in the proper sequence
  return df

def ParseExcelData(directory : str,
                    indexes : range, columns : list[str]
              ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
  """ Parsing Excel data from Battery Testing Machine

  Args:
      directory (str): Dataset directory location. !! Make sure not other file
  formats stored. No check has been added.
      indexes (range): The data indexes to select regions
      columns (list[str]): List of columns in String format.

  Returns:
      tuple[list[pd.DataFrame], list[pd.DataFrame]]: Data itself and SoC in the
  list format to capture absolutely all samples.
  """
  for _, _, files in os.walk(directory):
    files.sort(key=lambda f: int(f[-13:-5])) # Sort by last dates
    files = [directory + '/' + file for file in files]
    #! Running list comprehension 39.49
    # data_df : list[pd.DataFrame] = [
    #         Read_Excel_File(file) for file in files
    #     ]
    # data_SoC : list[pd.DataFrame] = [
    #         pd.DataFrame(
    #             data={'SoC' : diffSoC(
    #                 chargeData=df.loc[:,'Charge_Capacity(Ah)'].to_numpy(),
    #                 discargeData=df.loc[:,'Discharge_Capacity(Ah)'].to_numpy()
    #                 )},
    #             dtype=float32
    # 		) for df in data_df
    #     ]
    # return data_df, data_SoC

    #! Running threading (slower by 2 seconds) 42s
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #! Running multiprocesses (faster by 34 seconds) 8.61
    with concurrent.futures.ProcessPoolExecutor() as executor:
      data_df = list(executor.map(Read_Excel_File, files,
                    repeat(indexes), repeat(columns)))
      data_SoC = [
        pd.DataFrame(
            data={'SoC' : diffSoC(
                chargeData=df.loc[:,'Charge_Capacity(Ah)'],
                discargeData=df.loc[:,'Discharge_Capacity(Ah)']
                )},
            dtype=float32)
                for df in data_df]
      return data_df, data_SoC