from numpy import round, ndarray
from pandas import DataFrame, read_csv
# from numba import vectorize, jit

def str2bool(v : str) -> bool:
  """ Makes an input string to be a boolean variable.
  It will retur True to one of the following strings:
  
  @numba.jit has been removed since it only slows the process.
  
  Args:
    v (str): A boolean in string format.

  Returns:
    bool: True if [yes, true, True, t, 1], false - otherwise.
  """
  return v.lower() in ("yes", "true", "y", "t", "1")

# @vectorize
def diffSoC(chargeData : ndarray, discargeData : ndarray) -> ndarray:
  """ Round the SoC value to range of 0 to 1 with 2 decimal places by 
  subtracking Charge and Discharge

  Args:
      chargeData (Series): Charge_Capacity(Ah)
      discargeData (Series): Discharge_Capacity(Ah)

  Returns:
      ndarray: A rounded numpy array
  """
  return round((chargeData - discargeData)*100)/100

def Locate_Best_Epoch(file_path : str,
                      metric : str = 'val_root_mean_squared_error'
                    ):
  """ Reads the CSV file with history data and locates the smallest amongs 
  validation RMSE.

  Args:
      file_path (str): History file location
      metric (str): Column name to search by.
  Default 'val_root_mean_squared_error'.

  Returns:
      int: The index or the epoch number which had best result.
  """
  # df : DataFrame = read_csv(
  #           filepath_or_buffer=file_path, sep=",", delimiter=None,
  #           # Column and Index Locations and Names
  #           header="infer", names=None, index_col=None, usecols=None,
  #           prefix=None, mangle_dupe_cols=True,
  #           # General Parsing Configuration
  #           dtype=None, engine=None, converters=None, true_values=None,
  #           false_values=None, skipinitialspace=False, skiprows=None,
  #           skipfooter=0, nrows=None,
  #           # NA and Missing Data Handling
  #           na_values=None, keep_default_na=True, na_filter=True,
  #           verbose=False, skip_blank_lines=True,
  #           # Datetime Handling
  #           parse_dates=False, infer_datetime_format=False,
  #           keep_date_col=False, date_parser=None, dayfirst=False,
  #           cache_dates=True,
  #           # Iteration
  #           iterator=False, chunksize=None,
  #           # Quoting, Compression, and File Format
  #           compression="infer", thousands=None, decimal = ".",
  #           lineterminator=None, quotechar='"', quoting=0, doublequote=True,
  #           escapechar=None, comment=None, encoding=None, dialect=None,
  #           # Error Handling
  #           error_bad_lines=None, warn_bad_lines=None,
  #           # Internal
  #           delim_whitespace=False, low_memory=True, memory_map=False,
  #           float_precision=None
  #       ) #? FutureWarning: The warn_bad_lines, error_bad_lines argument has been deprecated and will be removed in a future version.
  #         #? FutureWarning: The squeeze argument has been deprecated and will be removed in a future version. Append .squeeze("columns") to the call to squeeze.
  df : DataFrame = read_csv(
            filepath_or_buffer=file_path, sep=","
        )
  #! Try catch to fix all files
  try:
    iEpoch : int = df['Epoch'][df[metric].idxmin()]
    value  : float = df[metric][df["Epoch"]==iEpoch].values[0]
    # except KeyError:
    #   print('>>>> NO EPOCH COLUMN')
    #   iEpoch : int = df[metric].idxmin()
    #   value  : float = df.iloc[iEpoch][metric]
    return (iEpoch, value)
  except TypeError:
    raise TypeError("Empty history")