from numpy import round, ndarray
from pandas import read_csv
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

def Locate_Best_Epoch(file_path : str) -> int:
  """ Reads the CSV file with history data and locates the smallest amongs 
  validation RMSE.

  Args:
      file_path (str): History file location

  Returns:
      int: The index or the epoch number which had best result.
  """
  return read_csv(
            filepath_or_buffer=file_path, sep=",", delimiter=None,
            # Column and Index Locations and Names
            header="infer", names=None, index_col=None, usecols=None,
            squeeze=False, prefix=None, mangle_dupe_cols=True,
            # General Parsing Configuration
            dtype=None, engine=None, converters=None, true_values=None,
            false_values=None, skipinitialspace=False, skiprows=None,
            skipfooter=0, nrows=None,
            # NA and Missing Data Handling
            na_values=None, keep_default_na=True, na_filter=True,
            verbose=False, skip_blank_lines=True,
            # Datetime Handling
            parse_dates=False, infer_datetime_format=False,
            keep_date_col=False, date_parser=None, dayfirst=False,
            cache_dates=True,
            # Iteration
            iterator=False, chunksize=None,
            # Quoting, Compression, and File Format
            compression="infer", thousands=None, decimal = ".",
            lineterminator=None, quotechar='"', quoting=0, doublequote=True,
            escapechar=None, comment=None, encoding=None, dialect=None,
            # Error Handling
            error_bad_lines=True, warn_bad_lines=True,
            # Internal
            delim_whitespace=False, low_memory=True, memory_map=False,
            float_precision=None
        )['val_root_mean_squared_error'].idxmin()