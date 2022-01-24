# %% [markdown]
# Supporting functions for determening SoC during data persering.
# %%
import pandas as pd
from scipy import integrate       # integration with trapizoid
import tensorflow as tf
#! Replance Tensorflow Numpy with numpy if version below 2.5
if(int(tf.__version__[2]) < 5):
    import numpy as tnp
else:
    import tensorflow.experimental.numpy as tnp

def diffSoC(chargeData   : pd.Series,
            discargeData : pd.Series) -> pd.Series:
  """ Return SoC based on differnece of Charge and Discharge Data.
  Data in range of 0 to 1.
  Args:
      chargeData (pd.Series): Charge Data Series
      discargeData (pd.Series): Discharge Data Series

  Raises:
      ValueError: If any of data has negative
      ValueError: If the data trend is negative. (end-beg)<0.

  Returns:
      pd.Series: Ceil data with 2 decimal places only.
  """
  # Raise error
  if((any(chargeData) < 0)
      |(any(discargeData) < 0)):
      raise ValueError("Parser: Charge/Discharge data contains negative.")
  #TODO: Finish up this check
  # if((chargeData[-1] - chargeData[0] < 0)
  #    |(discargeData[-1] - discargeData[0] < 0)):
  #     raise ValueError("Parser: Data trend is negative.")
  return tnp.round((chargeData - discargeData)*100)/100

def applyMinMax(data : pd.DataFrame) -> pd.Series:
  """ Uses Min-Max approach to shape SoC values withing range of 0-1

  Args:
      data (pd.DataFrame): SoC values in any form

  Returns:
      pd.Series: Rounded values of SoC in Percentage between 0-1
  """
  #!x' = (x-min_x)/(max_x-min_x)
  min : pd.Series = data.min()
  max : pd.Series = data.max()
  return tnp.round((data - min) / (max-min) * 100)/100
  
def ccSoC(current   : pd.Series,
        time_s    : pd.Series,
        n_capacity: float = 2.5 ) -> pd.Series:
  """ Return SoC based on Couloumb Counter.
  TODO: Ceil the data and reduce to 2 decimal places.

  Args:
      chargeData (pd.Series): Charge Data Series
      discargeData (pd.Series): Discharge Data Series

  Raises:
      ValueError: If any of data has negative
      ValueError: If the data trend is negative. (end-beg)<0.

  Returns:
      pd.Series: Ceil data with 2 decimal places only.
  """
  # Raise error
  if(any(time_s) < 0):
      raise ValueError("Parser: Time cannot be negative.")
  if(n_capacity == 0):
      raise ZeroDivisionError("Nominal capacity cannot be zero.")
  # Integration with trapezoid    
  # Nominal Capacity 2.5Ah. Double for that
  #uni_data_multi["CC"] = uni_data_multi["trapz(I)"]/3600*2.5
  #DoD = DsgCap - ChgCap
  #SOC = 1-Dod/Q
  #@ 25deg I said it was 2.5
  return (integrate.cumtrapz(current, time_s, initial=0)/abs(n_capacity*2))
