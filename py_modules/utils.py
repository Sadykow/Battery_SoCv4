from numpy import round, ndarray
from numba import vectorize

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

@vectorize
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