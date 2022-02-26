import sys
from tensorflow import Tensor, constant, round
from tensorflow.python.keras.models import Sequential, clone_model
from tensorflow.python.keras.layers import InputLayer, Dense
from tensorflow.python.ops.math_ops import exp

from typing import Callable
if (sys.version_info[1] < 9):
  LIST = list
  from typing import List as list
  from typing import Tuple as tuple
    
def create_ts_model(mFunc : Callable, layers : int = 1,
                 neurons : int = 500, dropout : float = 0.2,
                 input_shape : tuple = (500, 3), batch : int = 1
            ) -> Sequential:
  """ Creates Tensorflow 2 based time series models with inputs exception 
  handeling. Accepts multilayer models.
  TODO: For Py3.10 correct typing with: func( .. dropout : int | float .. )
  #!IMPORTANT: DO NOT pass tf.keras.layers.LSTM. Import and use LSTM instead
  #! ALSO, the cloning has to be done with following functions, not another
  #! import, due to TF implementation.
  Args:
    mFunc (Callable): Time series model function. .LSTM or .GRU
    layers (int, optional): № of layers. Above 1 will create a return
    sequence based models. Defaults to 1.
    neurons (int, optional): Total № of neurons across all layers.
    Value will be splitted evenly across all layers floored with 
    int() function. Defaults to 500.
    dropout (float, optional): Percentage dropout to eliminate random
    values. Defaults to 0.2.
    input_shape (tuple, optional): Input layer shape typle. Describes:
    (№_of_samles, №_of_deatues). Defaults to (500, 3).
    batch (int, optional): Batch size used at input layer. Defaults to 1.

  Raises:
    ZeroDivisionError: Rise an exception of anticipates unhandeled layer
    value, which cannot split neurons.

  Returns:
    tf.keras.models.Sequential: A sequentil model with single output and
    sigmoind() as an activation function.
  
  Examples:
    >>> lstm_model = create_ts_model(LSTM)
    >>> prec_model = clone_ts_model(LSTM)
  ```python
  #! NOTE: that if you pass tf.keras.layers.LSTM - function will create
  #! ModuleWrapper instead normal model. Some concept I do not comprehense
  from tensorflow.python.keras.layers import LSTM, GRU
  lstm_model = create_ts_model(
            LSTM, layers=nLayers, neurons=nNeurons,
            dropout=0.2, input_shape=x_train.shape[-2:], batch=1
        )
  prev_model = clone_cs_model(lstm_model)
  ```
  """
  # Check layers, neurons, dropout and batch are acceptable
  layers = 1 if layers == 0 else abs(layers)
  units : int = int(500/layers) if neurons == 0 else int(abs(neurons)/layers)
  dropout : float = float(dropout) if dropout >= 0 else float(abs(dropout))
  #? int(batch) if batch > 0 else ( int(abs(batch)) if batch != 0 else 1 )
  batch : int = int(abs(batch)) if batch != 0 else 1
  
  # Define sequential model with an Input Layer
  model : Sequential = Sequential([
          InputLayer(input_shape=input_shape,
                              batch_size=1)
      ])
  
  # Fill the layer content
  if(layers > 1): #* Middle connection layers
      for _ in range(layers-1):
          model.add(mFunc(
                  units=units, activation='tanh',
                  dropout=dropout, return_sequences=True
              ))
  if(layers > 0): #* Last no-connection layer
      model.add(mFunc(
              units=units, activation='tanh',
              dropout=dropout, return_sequences=False
          ))
  else:
      print("Unhaldeled exeption with Layers")
      raise ZeroDivisionError
  
  # Define the last Output layer with sigmoind
  model.add(Dense(
          units=1, activation='sigmoid', use_bias=True
      ))
  
  # Return completed model with some info if neededs
  # print(model.summary())
  return model

def clone_cs_model(model):
  return clone_model(model, input_tensors=None, clone_function=None)

def tf_round(x : Tensor, decimals : int = 0) -> Tensor:
  """ Round to nearest decimal

  Args:
      x (tf.Tensor): Input value or array
      decimals (int, optional): How many precisions. Defaults to 0.

  Returns:
      tf.Tensor: Return rounded value
  """
  multiplier : Tensor = constant(
          value=10**decimals, dtype=x.dtype, shape=None,
          name='decimal_multiplier'
      )
  return round(
              x=(x * multiplier), name='round_to_decimal'
          ) / multiplier

def scheduler(epoch : int, lr : float) -> float:
  """ Scheduler
  round(model.optimizer.lr.numpy(), 5)

  Args:
      epoch (int): [description]
      lr (float): [description]

  Returns:
      float: [description]
  """
  #! Think of the better sheduler
  if (epoch < 20):
      return lr
  else:
      # lr = tf_round(x=lr * tf.math.exp(-0.05), decimals=6)
      lr = lr * exp(-0.05)
      if lr >= 0.00005:
          return lr
      else:
          return  0.00005
  # return np.arange(0.001,0,-0.00002)[iEpoch]
  # return lr * 1 / (1 + decay * iEpoch)

def get_learning_rate(epoch : int, iLr : float) -> float:
  """_summary_

  Args:
      epoch (int): _description_
      iLr (float): _description_

  Returns:
      float: _description_
  """
  for i in range(0, epoch):
    iLr = scheduler(i, iLr)
  print(f'The Learning rate set to: {iLr}')
  return iLr