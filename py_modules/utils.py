from pandas import Series
from numpy import round, ndarray
def str2bool(v : str) -> bool:
  """ Makes an input string to be a boolean variable.
  It will retur True to one of the following strings:
  
  
  Args:
    v (str): A boolean in string format.

  Returns:
    bool: True if [yes, true, True, t, 1], false - otherwise.
  """
  return v.lower() in ("yes", "true", "y", "t", "1")

def diffSoC(chargeData : Series, discargeData : Series) -> ndarray:
  """ Round the SoC value to range of 0 to 1 with 2 decimal places by 
  subtracking Charge and Discharge

  Args:
      chargeData (Series): Charge_Capacity(Ah)
      discargeData (Series): Discharge_Capacity(Ah)

  Returns:
      ndarray: A rounded numpy array
  """
  return round((chargeData.to_numpy() - discargeData.to_numpy())*100)/100