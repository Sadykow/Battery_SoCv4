# %% [markdown]
# Attention mechanism implemented by Genta Indra Winata
# Github: gentaiscool
# https://github.com/gentaiscool/lstm-attention
# %%
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
############################################## 
"""
# ATTENTION LAYER
Cite these works 
1. Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
"Hierarchical Attention Networks for Document Classification"
accepted in NAACL 2016
2. Winata, et al. https://arxiv.org/abs/1805.12307
"Attention-Based LSTM for Psychological Stress Detection from Spoken Language Using Distant Supervision." 
accepted in ICASSP 2018

Using a context vector to assist the attention

* How to use:
Put return_sequences=True on the top of an RNN Layer (GRU/LSTM/SimpleRNN).
The dimensions are inferred based on the output shape of the RNN.

Example:
  model.add(LSTM(64, return_sequences=True))
  model.add(AttentionWithContext())
  model.add(Addition())
  # next add a Dense layer (for classification/regression) or whatever...
"""
##############################################

def dot_product(x, kernel):
  """
  Wrapper for dot product operation, in order to be compatible with both
  Theano and Tensorflow
  Args:
    x (): input
    kernel (): weights
  Returns:
  """
  print(f'DotProduct X: {x} or type: {type(x)}')
  print(f'DotProduct kernel: {kernel} or type: {type(kernel)}')

  if tf.keras.backend.backend() == 'tensorflow':
    # return tnp.squeeze(
    #             a=tnp.dot(
    #                 a=x,
    #                 b=tnp.expand_dims(
    #                     a=kernel,
    #                     axis=0
    #                   )
    #               ),
    #             axis=-1)
    return tf.keras.backend.squeeze(x=tf.keras.backend.dot(
                                      x=x,
                                      y=tf.keras.backend.expand_dims(x=kernel,
                                                                     axis=-1)),
                                    axis=-1)
  else:
    return tnp.dot(a=x, b=kernel)
# %%
class AttentionWithContext(tf.keras.layers.Layer):
  """
  Attention operation, with a context/query vector, for temporal data.
  Supports Masking.

  follows these equations:
  
  (1) u_t = tanh(W h_t + b)
  (2) \alpha_t = \frac{exp(u^T u)}{\sum_t(exp(u_t^T u))}, this is the attention weight
  (3) v_t = \alpha_t * h_t, v in time t

  # Input shape
    3D tensor with shape: `(samples, steps, features)`.
  # Output shape
    3D tensor with shape: `(samples, steps, features)`.

  """
  supports_masking : bool
  init  : tf.keras.initializers.GlorotUniform

  W_regularizer : tf.keras.regularizers.Regularizer
  u_regularizer : tf.keras.regularizers.Regularizer
  b_regularizer : tf.keras.regularizers.Regularizer

  W_constraint  : tf.keras.constraints.Constraint
  u_constraint  : tf.keras.constraints.Constraint
  b_constraint  : tf.keras.constraints.Constraint

  bias  : bool

  local_name  : str
  def __init__(self,
         W_regularizer=None, u_regularizer=None, b_regularizer=None,
         W_constraint=None, u_constraint=None, b_constraint=None,
         bias=True, **kwargs):

    self.supports_masking = True
    self.init = tf.keras.initializers.get('glorot_uniform')

    self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
    self.u_regularizer = tf.keras.regularizers.get(u_regularizer)
    self.b_regularizer = tf.keras.regularizers.get(b_regularizer)

    self.W_constraint = tf.keras.constraints.get(W_constraint)
    self.u_constraint = tf.keras.constraints.get(u_constraint)
    self.b_constraint = tf.keras.constraints.get(b_constraint)

    self.bias = bias
    super(AttentionWithContext, self).__init__(**kwargs)

  def build(self, input_shape):
    assert len(input_shape) == 3
    self.local_name = 'atten'
    self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                 initializer=self.init,
                 name='{}_W'.format(self.local_name),
                 regularizer=self.W_regularizer,
                 constraint=self.W_constraint)
    if self.bias:
      self.b = self.add_weight(shape=(input_shape[-1],),
                   initializer='zero',
                   name='{}_b'.format(self.local_name),
                   regularizer=self.b_regularizer,
                   constraint=self.b_constraint)

    self.u = self.add_weight(shape=(input_shape[-1],),
                 initializer=self.init,
                 name='{}_u'.format(self.local_name),
                 regularizer=self.u_regularizer,
                 constraint=self.u_constraint)
    print(f'Build W: {self.W} or type: {type(self.W)}')
    print(f'Build b: {self.b} or type: {type(self.b)}')
    print(f'Build u: {self.u} or type: {type(self.u)}')
  
    super(AttentionWithContext, self).build(input_shape)

  def compute_mask(self, input, input_mask=None) -> None:
    # do not pass the mask to the next layers
    return None

  def call(self, x, mask=None):
    print(f'Call x: {x} or type: {type(x)}')
    print(f'Call mask: {mask} or type: {type(mask)}')
    uit = dot_product(x, self.W)

    if self.bias:
      uit += self.b

    uit = tf.keras.backend.tanh(uit)
    ait = dot_product(uit, self.u)

    a = tf.keras.backend.exp(ait)

    # apply mask after the exp. will be re-normalized next
    if mask is not None:
      # Cast the mask to floatX to avoid float64 upcasting in theano
      a *= tf.keras.backend.cast(mask, tf.keras.backend.floatx())

    # in some cases especially in the early stages of training the sum may be almost zero and this results in NaN's. 
    # Should add a small epsilon as the workaround
    # a /= tf.keras.backend.cast(tf.keras.backend.sum(a, axis=1, keepdims=True), tf.keras.backend.floatx())
    a /= tf.keras.backend.cast(tf.keras.backend.sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(), tf.keras.backend.floatx())

    a = tf.keras.backend.expand_dims(a)
    weighted_input = x * a
    
    return weighted_input

  def compute_output_shape(self, input_shape):
    print(f'ComputeOutputShape x: {input_shape} or type: {type(input_shape)}')
    return input_shape[0], input_shape[1], input_shape[2]
#AttentionWithContext.build = build
# %%	
class Addition(tf.keras.layers.Layer):
  """
  This layer is supposed to add of all activation weight.
  We split this from AttentionWithContext to help us getting the activation weights

  follows this equation:

  (1) v = \sum_t(\alpha_t * h_t)
  
  # Input shape
    3D tensor with shape: `(samples, steps, features)`.
  # Output shape
    2D tensor with shape: `(samples, features)`.
  """
  output_dim  : int

  def __init__(self, **kwargs):
    super(Addition, self).__init__(**kwargs)

  def build(self, input_shape):
    self.output_dim = input_shape[-1]
    super(Addition, self).build(input_shape)

  def call(self, x):
    return tf.keras.backend.sum(x, axis=1)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)