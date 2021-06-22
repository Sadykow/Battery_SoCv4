# %% [markdown]
# # Auto-Regression
# Implementation based on Feedback example from Time Series example.
# %%
import tensorflow as tf
import numpy as np

class AutoFeedBack(tf.keras.Model):
  out_steps   : int
  num_feat    : int
  float_dtype : type = tf.float64

  gru_cell    : tf.keras.layers.GRUCell
  gru_rnn     : tf.keras.layers.RNN
  dense       : tf.keras.layers.Dense
  
  units       : int
  def __init__(self, units : int, out_steps : int, num_features : int = 1,
              float_dtype : type = tf.float32, **kwargs):
    super(AutoFeedBack, self).__init__(**kwargs)
    self.out_steps      = out_steps
    self.num_feat       = num_features
    self.float_dtype    = float_dtype
    self.units          = units
    #(tf.keras.Input(shape=(x,500,4)))
    self.gru_cell = tf.keras.layers.GRUCell(units=self.units,
        activation='tanh', recurrent_activation='sigmoid', use_bias=True,
        kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
        bias_initializer='zeros', kernel_regularizer=None,
        recurrent_regularizer=None, bias_regularizer=None,
        kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
        dropout=0.0, recurrent_dropout=0, reset_after=True
      )

    # Also wrap the GRUCell in an RNN to simplify the `warmup` method.
    self.gru_rnn = tf.keras.layers.RNN(self.gru_cell,
        return_sequences=False, return_state=True,
        go_backwards=False, stateful=False, unroll=False, time_major=False
      )
    # Set the output layer
    self.dense = tf.keras.layers.Dense(units=num_features,
        activation='sigmoid', use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None
      )
  
  def warmup(self, inputs):
    # print(type(inputs))
    # inputs.shape => (batch, time, features)
    # x.shape => (batch, lstm_units)
    # x, *state = self.gru_rnn(inputs[:,:-self.out_steps,:])
    x, *state = self.gru_rnn(inputs)

    # predictions.shape => (batch, features)
    prediction = self.dense(x)
    return prediction, state

  @tf.function
  def call(self, inputs, training=None):
    if training:
      # Use a TensorArray to capture dynamically unrolled outputs.
      predictions = tf.TensorArray(self.float_dtype, size=self.out_steps,
                dynamic_size=False, clear_after_read=True,
                tensor_array_name=None, handle=None, flow=None,
                infer_shape=True, element_shape=tf.TensorShape([1, None]),
                colocate_with_first_write_call=True, name=None)
      # Initialize the lstm state # -> (1, 492, 4)
      prediction, state = self.warmup(inputs[:,:-self.out_steps,:])
      predictions = predictions.write(0, prediction)
      
      # Run the rest of the prediction steps
      for n in range(1, self.out_steps):
        # Use the last prediction as input.
        x = tf.concat(
                    values=[
                        inputs[:,-self.out_steps+n,:-self.num_feat],
                        prediction
                        ],
                    axis=1
                )
        # Execute one lstm step.
        x, state = self.gru_cell(
                x,
                states=state,
                training=training)
        # Convert the lstm output to a prediction.
        prediction = self.dense(x)
        # Add the prediction to the output
        predictions = predictions.write(n, prediction)

      # predictions.shape => (time, batch, features)
      return predictions.stack()[:,0,0]
    else:
      x, *_ = self.gru_rnn(inputs, training=training)
      predictions = self.dense(x)
      return self.tf_round(predictions[0], decimals=2)

  def tf_round(self, x : tf.Tensor, decimals : int = 2):
    #multiplier = tf.constant(10**decimals, dtype=x.dtype)
    #return tf.round(x * multiplier) / multiplier
    multiplier  : tf.Tensor = tf.constant(value = 10**decimals, shape=(1,),
                        dtype=x.dtype, name = 'Multiplier')
    return tf.math.divide(
                x=tf.keras.backend.round(
                      tf.math.multiply(
                          x=x,
                          y=multiplier
                        )
                    ),
                y=multiplier,
                name = 'Convert_Back'
              )
  
  def __repr__(self) -> str:
    return super().__repr__()

  def get_config(self):
    return {"units": self.units,
            "out_steps" : self.out_steps}

  @classmethod
  def from_config(cls, config):
      return cls(**config)

    
