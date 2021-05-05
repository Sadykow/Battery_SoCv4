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

  lstm_cell   : tf.keras.layers.LSTMCell
  lstm_rnn    : tf.keras.layers.RNN
  dense       : tf.keras.layers.Dense
  
  def __init__(self, units : int, out_steps : int, num_features : int = 1,
              float_dtype : type = tf.float32):
    super().__init__()
    self.out_steps      = out_steps
    self.num_feat       = num_features
    self.float_dtype    = float_dtype
    #(tf.keras.Input(shape=(x,500,4)))
    self.lstm_cell = tf.keras.layers.LSTMCell(units,
        activation='tanh', recurrent_activation='sigmoid', use_bias=True,
        kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
        bias_initializer='zeros', unit_forget_bias=True,
        kernel_regularizer=None, recurrent_regularizer=None,
        bias_regularizer=None, kernel_constraint=None,
        recurrent_constraint=None, bias_constraint=None, dropout=0.0, #!0.2
        recurrent_dropout=0
      )
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell,
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
    # x, *state = self.lstm_rnn(inputs[:,:-self.out_steps,:])
    x, *state = self.lstm_rnn(inputs)

    # predictions.shape => (batch, features)
    prediction = self.dense(x)
    return prediction, state

  # def call(self, inputs, training=None):
  #   # Capture dynamically unrolled outputs.
  #   predictions : tf.TensorArray = tf.TensorArray(dtype=self.float_dtype,
  #       size=self.out_steps, dynamic_size=False, clear_after_read=False,
  #       tensor_array_name=None, handle=None, flow=None, infer_shape=True,
  #       element_shape=None, colocate_with_first_write_call=True,
  #       name='predictions'
  #     )
  #   i_pred : int = 0
  #   # Initialize the lstm state
  #   prediction, state = self.warmup(inputs)

  #   # Insert the first prediction
  #   predictions = predictions.write(i_pred, prediction)
  #   i_pred+=1

  #   # Run the rest of the prediction steps
  #   for n in range(i_pred, self.out_steps):
  #     # Use the last prediction as input.
  #     # Execute one lstm step.
  #     x, state = self.lstm_cell(
  #               tf.concat(
  #                   values=[
  #                       inputs[:,-self.out_steps+n,:-self.num_feat],
  #                       prediction
  #                     ],
  #                   axis=1,
  #                   name='InputPred'
  #                 ),
  #               states=state,
  #               training=training)
  #     # Convert the lstm output to a prediction.
  #     prediction = self.dense(x)
  #     # Add the prediction to the output
  #     predictions = predictions.write(n, prediction)

  #   # predictions.stack().shape     => (time, batch, features)
  #   # predictions.transpose().shape => (batch, time, features)
  #   return tf.transpose(
  #               a = predictions.stack(),
  #               perm = [1, 0, 2],
  #               conjugate = False,
  #               name = 'TranposeOrder'
  #             )

  def call(self, inputs, training=None):
    ###Given model with 510cells, out_steps=8, Dense(1) ###
    # print(f'\nTraining State: {training}')
    # print(f"Input Shape: {inputs.shape}") # -> (1, 500, 4)
    if training:
      # Use a TensorArray to capture dynamically unrolled outputs.
      predictions = []
      # Initialize the lstm state # -> (1, 492, 4)
      # print(f"First Input Shape: {inputs[:,:-self.out_steps,:].shape}") 
      prediction, state = self.warmup(inputs[:,:-self.out_steps,:])
      # print(f"Warmup Pred[0]:  {prediction[0].shape}") #-> (1,)
      # print(f"Warmup Len State: {len(state)}") #-> 2
      # for i in range(len(state)):
        # print(f"Warmup State[{i}]: {state[i].shape}")
        # -> 2 -> (1,510)
        # Insert the first prediction
      predictions.append(prediction)

      # Run the rest of the prediction steps
      for n in range(1, self.out_steps):
        # Use the last prediction as input.
        # x = tf.concat(
        #             values=[
        #                 inputs[:,-self.out_steps:-self.out_steps+n,:-self.num_feat],
        #                 tf.expand_dims(prediction, axis=0)
        #                 ],
        #             axis=2,
        #             name='InputPred'
        #             )
        x = tf.concat(
                    values=[
                        inputs[:,-self.out_steps+n,:-self.num_feat],
                        prediction
                        ],
                    axis=1,
                    name='InputPred'
                )
        # print(f"Concated: {x.shape}")
        # Execute one lstm step.
        x, state = self.lstm_cell(
                x,
                states=state,
                training=training)
        # Convert the lstm output to a prediction.
        prediction = self.dense(x)
        # Add the prediction to the output
        predictions.append(prediction)

      # predictions.shape => (time, batch, features)
      predictions = tf.stack(predictions)
      # print(f'Stacked: {predictions.shape}')
      # predictions.shape => (batch, time, features)
      # predictions = tf.transpose(predictions, [1, 0, 2])
      # print(f'Transposed: {predictions.shape}')
      # print(f'Returned: {predictions[:,0,0].shape}')
      return predictions[:,0,0]
      #return self.tf_round(predictions[:,0,0], decimals=2)
    else:
      x, *_ = self.lstm_rnn(inputs, training=training)
      predictions = self.dense(x)
      # print(f'Validation shape: {predictions[0].shape}')
      return self.tf_round(predictions[0], decimals=2)
  # AutoFeedBack.call = call
  
  def call(self, inputs, training=None):
    ###Given model with 510cells, out_steps=8, Dense(1) ###
    # print(f'\nTraining State: {training}')
    # print(f"Input Shape: {inputs.shape}") # -> (1, 500, 4)
    if training:
      # Use a TensorArray to capture dynamically unrolled outputs.
      predictions = []
      # Initialize the lstm state # -> (1, 492, 4)
      # print(f"First Input Shape: {inputs[:,:-self.out_steps,:].shape}") 
      prediction, state = self.warmup(inputs[:,:-self.out_steps,:])
      # print(f"Warmup Pred[0]:  {prediction[0].shape}") #-> (1,)
      # print(f"Warmup Len State: {len(state)}") #-> 2
      # for i in range(len(state)):
        # print(f"Warmup State[{i}]: {state[i].shape}")
        # -> 2 -> (1,510)
        # Insert the first prediction
      predictions.append(prediction)

      # Run the rest of the prediction steps
      for n in range(1, self.out_steps):
        # Use the last prediction as input.
        # x = tf.concat(
        #             values=[
        #                 inputs[:,-self.out_steps:-self.out_steps+n,:-self.num_feat],
        #                 tf.expand_dims(prediction, axis=0)
        #                 ],
        #             axis=2,
        #             name='InputPred'
        #             )
        x = tf.concat(
                    values=[
                        inputs[:,-self.out_steps+n,:-self.num_feat],
                        prediction
                        ],
                    axis=1,
                    name='InputPred'
                )
        # print(f"Concated: {x.shape}")
        # Execute one lstm step.
        x, state = self.lstm_cell(
                x,
                states=state,
                training=training)
        # Convert the lstm output to a prediction.
        prediction = self.dense(x)
        # Add the prediction to the output
        predictions.append(prediction)

      # predictions.shape => (time, batch, features)
      predictions = tf.stack(predictions)
      # print(f'Stacked: {predictions.shape}')
      # predictions.shape => (batch, time, features)
      # predictions = tf.transpose(predictions, [1, 0, 2])
      # print(f'Transposed: {predictions.shape}')
      # print(f'Returned: {predictions[:,0,0].shape}')
      return predictions[:,0,0]
      #return self.tf_round(predictions[:,0,0], decimals=2)
    else:
      x, *_ = self.lstm_rnn(inputs, training=training)
      predictions = self.dense(x)
      # print(f'Validation shape: {predictions[0].shape}')
      return self.tf_round(predictions[0], decimals=2)

  def tf_round(self, x : tf.Tensor, decimals : int = 2):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

  def __repr__(self) -> str:
    return super().__repr__()