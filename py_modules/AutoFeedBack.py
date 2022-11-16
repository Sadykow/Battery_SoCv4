# %% [markdown]
# # Auto-Regression
# Implementation based on Feedback example from Time Series example.
# %%
import tensorflow as tf

class AutoFeedBack(tf.keras.Model):
  out_steps   : int
  num_feat    : int
  float_dtype : type = tf.float64

  lstm_cell   : tf.keras.layers.LSTMCell
  lstm_rnn    : tf.keras.layers.RNN
  dense       : tf.keras.layers.Dense
  
  units       : int
  #TODO: Introduce Save/Load
  def __init__(self, units : int, out_steps : int, num_features : int = 1,
              float_dtype : type = tf.float32):
    """ Model cinstructor initialising layers. This one utilises LSTM only.
    Although the cell type can be changed or passed as a parameter.

    Args:
        units (int): № of units in the LSTM cell
        out_steps (int): № of output steps
        num_features (int, optional): This parameter defines how many features
        expected to be outputted. Usefull in multi output models. Defaults to 1.
        float_dtype (type, optional): Varaibles types to use.
        Defaults to tf.float32.
    """
    super().__init__()
    self.out_steps      = out_steps
    self.num_feat       = num_features
    self.float_dtype    = float_dtype
    self.units          = units
    #(tf.keras.Input(shape=(x,500,4)))
    self.lstm_cell = tf.keras.layers.LSTMCell(units=self.units,
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
    """ Warmup procedure. Runs initial procedure of prediction.

    Args:
        inputs (tf.Array): Input samples

    Returns:
        tuple: Prediction result and state of the model to reuse it again.
    """
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

  def call_noTensor(self, inputs, training=None):
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
  
  @tf.function
  def call(self, inputs, training=None):
    """ Primary call method to iterate over the training. Two separate ways
    implements different procedures for training and testing. Testing procedure
    no different than any regular predictions

    Args:
        inputs (tf.Array): Input feature samples with hostiry samples
        training (bool, optional): Separates procedure between training/fit
        and testing/prediction. Defaults to None.

    Returns:
        tf.Array: Final array of prediction or predictions of out_steps.
    """
    if training:
      # Use a TensorArray to capture dynamically unrolled outputs.
      predictions = tf.TensorArray(self.float_dtype, size=self.out_steps,
                dynamic_size=False, clear_after_read=True,
                tensor_array_name=None, handle=None, flow=None,
                infer_shape=True, element_shape=tf.TensorShape([1, None]),
                colocate_with_first_write_call=True, name=None)
      # Initialize the lstm state # -> (1, 492, 4)
      # print(f"First Input Shape: {inputs[:,:-self.out_steps,:].shape}") 
      prediction, state = self.warmup(inputs[:,:-self.out_steps,:])
      # print(f"Warmup Pred[0]:  {prediction[0].shape}") #-> (1,)
      # print(f"Warmup Len State: {len(state)}") #-> 2
      # for i in range(len(state)):
        # print(f"Warmup State[{i}]: {state[i].shape}")
        # -> 2 -> (1,510)
        # Insert the first prediction
      predictions = predictions.write(0, prediction)
      
      # Run the rest of the prediction steps
      # for n in range(1, self.out_steps):
      for n in tf.range(1, self.out_steps, delta=1,
                        dtype=tf.int32, name='range'):
        # Use the last prediction as input.
        x = tf.concat(
                    values=[
                        inputs[:,-self.out_steps+n,:-self.num_feat],
                        prediction
                        ],
                    axis=1
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
        predictions = predictions.write(n, prediction)

      # predictions.shape => (time, batch, features)
      # return tf.squeeze(tf.squeeze(predictions.stack()))
      return predictions.stack()[:,0,0]
      # return self.tf_gradient_round(predictions.stack()[:,0,0], 2)
    else:
      x, *_ = self.lstm_rnn(inputs, training=training)
      predictions = self.dense(x)
      # print(f'Validation shape: {predictions[0].shape}')
      return self.tf_round(predictions[0], decimals=2)

  @tf.function(experimental_compile=True)
  def call_while(self, inputs, training=None):
    """ Primary call method to iterate over the training. Two separate ways
    implements different procedures for training and testing. Testing procedure
    no different than any regular predictions.
    This version relies on the while loop implementation.

    Args:
        inputs (tf.Array): Input feature samples with hostiry samples
        training (bool, optional): Separates procedure between training/fit
        and testing/prediction. Defaults to None.

    Returns:
        tf.Array: Final array of prediction or predictions of out_steps.
    """
    if training:
      # Use a TensorArray to capture dynamically unrolled outputs.
      # predictions = tf.TensorArray(self.float_dtype, size=self.out_steps,
      #           dynamic_size=False, clear_after_read=False,
      #           tensor_array_name=None, handle=None, flow=None,
      #           infer_shape=True, element_shape=tf.TensorShape([1, None]),
      #           colocate_with_first_write_call=True, name=None)
      predictions = tf.Variable([0.0]*self.out_steps, dtype=self.float_dtype)
      # Initialize the lstm state # -> (1, 492, 4)
      # print(f"First Input Shape: {inputs[:,:-self.out_steps,:].shape}") 
      prediction, state = self.warmup(inputs[:,:-self.out_steps,:])
      # print(f"Warmup Pred:  {prediction.shape}") #-> (1,)
      # for i in range(len(prediction)):
      #   print(f"Warmup Pred[{i}]: {prediction[i].shape}")
      #   # print(f"Warmup Pred[0]:  {prediction[0].shape}") #-> (1,)
      
      # try:
      #   print(f"Warmup Len State: {len(state)}") #-> 2
      #   for i in range(len(state)):
      #     print(f"Warmup State[{i}]: {state[i].shape}")
      #     # -> 2 -> (1,510)
      # except Exception as e:
      #   print("Failed to get len state")
      
      # Insert the first prediction
      n = tf.constant(0)
      # predictions = predictions.write(n, prediction)
      predictions = tf.concat(values=[predictions, prediction[0]], axis=0)
      predictions = predictions[1:]
      # n = 0
      # print(f"Counter: {n.get_shape()}")
      tf.add(x=n,y=1,name='condition')
      # n += 1
      # print(f"Counter+1: {n.get_shape()}")
      # print(f'Out_steps: {self.out_steps}')
      # print(f'Num_feataures: {self.num_feat}')
      # cond = lambda i: tf.less(x=n, y=self.out_steps)
      def cond(n, out_steps, *args):
        return tf.less(x=n, y=out_steps)
    
      def inner_loop(n : tf.Tensor, out_steps : int, num_feat : int,
                     inputs : tf.TensorArray, prediction, state,
                     predictions : tf.TensorArray):
        # Use the last prediction as input.
        x = tf.concat(
                    values=[
                        inputs[:,-out_steps+n,:-num_feat],
                        prediction
                        ],
                    axis=1
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
        # predictions = predictions.write(n, prediction)
        predictions = tf.concat(values=[predictions, prediction[0]], axis=0)
        predictions = predictions[1:]
        # print(f">Warmup Pred:  {prediction.shape}") #-> (1,)
        # for i in range(len(predictions)):
        #   print(f">Warmup Pred[{i}]: {prediction[i].shape}")
        #   # print(f"Warmup Pred[0]:  {prediction[0].shape}") #-> (1,)
        
        # try:
        #   print(f">>Warmup Len State: {len(state)}") #-> 2
        #   for i in range(len(state)):
        #     print(f">>Warmup State[{i}]: {state[i].shape}")
        # except Exception as e:
        #   print(">>Failed to get len state")

        # print(f'Input shape: {inputs.shape}')
        # print(f'Input type: {type(inputs)}')
        # print(f"Counter: {n.get_shape()}")
        tf.add(x=n,y=1,name='condition')
        # n += 1
        # print(f"Counter+1: {n.get_shape()}")
        # print(f'Out_steps: {out_steps}')
        # print(f'Num_feataures: {num_feat}')
        return n, out_steps, num_feat, inputs, prediction, state, predictions

      # print(f'Input type: {type(inputs)}')
      # _, _, _, _, _, _, result = tf.nest.map_structure(
      #   tf.stop_gradient,
      #   tf.while_loop(
      #     cond=cond,
      #     body=inner_loop,
      #     loop_vars=[n, self.out_steps, self.num_feat,
      #               inputs, prediction, state,
      #               predictions],
      #     shape_invariants=None,
      #     parallel_iterations=10,
      #     back_prop=True, swap_memory=False, maximum_iterations=self.out_steps,
      #     name='AutoFeedBack'
      #   )
      # )
      _, _, _, _, _, _, result = tf.while_loop(
          cond=cond,
          body=inner_loop,
          loop_vars=[n, self.out_steps, self.num_feat,
                    inputs, prediction, state,
                    predictions],
          shape_invariants=[n.get_shape(),n.get_shape(),n.get_shape(),
                    inputs.shape, prediction.shape,[state[0].shape,state[0].shape], predictions.get_shape()],
          parallel_iterations=10,
          back_prop=True, swap_memory=False, maximum_iterations=self.out_steps,
          name='AutoFeedBack'
        )

      # return result.stack()[:,0,0]
      return result

    else:
      x, *_ = self.lstm_rnn(inputs, training=training)
      predictions = self.dense(x)
      # print(f'Validation shape: {predictions[0].shape}')
      return self.tf_round(predictions[0], decimals=2)

  @tf.function
  def tf_round(self, x : tf.Tensor, decimals : int = 2):
    """ Direct rounding, similar to the simple tf.round. Resets gradient.

    Args:
        x (tf.Tensor): Array or varaible to round
        decimals (int, optional): Number pf decimals to preserve. Defaults to 2.

    Returns:
        tf.Variable: Rounded output
    """
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
  
  def tf_gradient_round(self, x : tf.Tensor, decimals : int = 2):
    """ Output rounding without overwriting the gradient. Intedned for testing
    uses only, but acceptable for training as well. Usual tf.round() will reset
    gradients, which will interupt the training process.

    Args:
        x (tf.Tensor): [description]
        decimals (int, optional): [description]. Defaults to 2.

    Returns:
        tf.Variable: Rounded output
    """
    #* *100
    multiplier  : tf.Tensor = tf.constant(value = 10**decimals, shape=(1,),
                        dtype=x.dtype, name = 'Multiplier')
    #* 0.9356 -> 93.56
    raised = tf.math.multiply(x, multiplier)
    #* 93.56 -> 93.00
    rounded_pos = tf.keras.backend.cast(
                        tf.keras.backend.cast(
                            raised,
                            dtype=tf.int32
                          ),
                        dtype=x.dtype
                      )
    #* 93.56-93.00 = 0.56 
    cutter = tf.math.subtract(
                      raised,
                      rounded_pos
                    )
    #* 93.56-0.56->93.00->0.93
    prediction = tf.math.divide(
                      tf.math.subtract(
                            raised,
                            cutter
                          ),
                      multiplier
                    )
    #*+1 or 0
    return prediction + tf.round(cutter)/multiplier
  
  def __repr__(self) -> str:
    return super().__repr__()

  def get_config(self):
    """ Config preservation for model saving and restoring.

    Returns:
        Dictinory: config variables, which is not defined by parent class.
    """
    return {"units": self.units,
            "out_steps" : self.out_steps}

  @classmethod
  def from_config(cls, config):
      return cls(**config)
    
