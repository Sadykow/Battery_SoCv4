# %% [markdown]
# # Auto-Regression
# Implementation based on Feedback example from Time Series example.
# %%
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

class AutoFeedBack(tf.keras.Model):
  out_steps : int
  num_features  : int

  lstm_cell : tf.keras.layers.LSTMCell

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def __repr__(self) -> str:
    return super().__repr__()