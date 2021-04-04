def str2bool(v : str) -> bool:
  """ Makes an input string to be a boolean variable.
  It will retur True to one of the following strings:
  
  
  Args:
    v (str): A boolean in string format.

  Returns:
    bool: True if [yes, true, True, t, 1], false - otherwise.
  """
  return v.lower() in ("yes", "true", "y", "t", "1")