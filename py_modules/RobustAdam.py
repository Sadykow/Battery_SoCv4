from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.util.tf_export import keras_export
import tensorflow as tf
from numpy import float32

import sys
if (sys.version_info[1] < 9):
  LIST = list
  from typing import List as list

# @keras_export('tf.keras.optimizers.RobustAdam')
class RobustAdam(tf.keras.optimizers.Optimizer):
  
  epsilon : tf.float32 = 10e-8
  
  # Robust part
  prev_loss : tf.Variable = None
  current_loss : tf.Variable = None

  def __init__(self, name : str ='RobustAdam',
                learning_rate: tf.float32 = 0.0001,
                beta_1 : tf.float32 = 0.9,
                beta_2 : tf.float32 = 0.999,
                beta_3 : tf.float32 = 0.999,
                k      : tf.float32 = 0.1,
                K      : tf.float32 = 10,
                epsilon :tf.float32 = 1e-7,
                **kwargs):
    super(RobustAdam, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self._set_hyper('beta_3', beta_3)
    self._set_hyper('k', k)
    self._set_hyper('K', K)
    self.epsilon = epsilon # or backend_config.epsilon()
    
  def _create_slots(self, var_list : list[tf.Variable]) -> None:
    """For each model variable, create the optimizer variable associated
    with it. TensorFlow calls these optimizer variables "slots".
    For momentum optimization, we need one momentum slot per model variable.

    Create slots for the first, second and third moments.
    Separate for-loops to respect the ordering of slot variables from v1.

    Args:
        var_list (list[tf.Variable]): List of varaibles to add slots into
    """
    for var in var_list:
      self.add_slot(var, slot_name='m', initializer="zeros")
    for var in var_list:
      self.add_slot(var, slot_name='v', initializer='zeros')
    for var in var_list:
      self.add_slot(var, slot_name='d', initializer='ones')
    # for var in var_list:
    #   self.add_slot(var, slot_name='prev_loss',
    #                      initializer=self.prev_loss)
    # for var in var_list:
    #   self.add_slot(var, slot_name='current_loss', 
    #                      initializer=self.current_loss)
    # print('_create_slots')
    
  def _prepare_local(self, var_device, var_dtype, apply_state) -> None:
    """ Preparing varaibles locally. Initialising some of the states to match
    the algorithm.

    Args:
        var_device (): device
        var_dtype (): type
        apply_state (): function*
    """
    super(RobustAdam, self)._prepare_local(var_device, var_dtype, apply_state)
    local_step = tf.cast(self.iterations + 1, var_dtype)
    beta_1_t = self._get_hyper('beta_1', var_dtype)
    beta_2_t = self._get_hyper('beta_2', var_dtype)
    beta_3_t = self._get_hyper('beta_3', var_dtype)
    k_t      = self._get_hyper('k', var_dtype)
    K_t      = self._get_hyper('K', var_dtype)
    beta_1_power = tf.pow(beta_1_t, local_step)
    beta_2_power = tf.pow(beta_2_t, local_step)
    lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
         (tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
    apply_state[(var_device, var_dtype)].update(
        dict(
            lr=lr,                
            epsilon=float32(self.epsilon),
            beta_1_t=beta_1_t,
            beta_1_power=beta_1_power,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            beta_2_power=beta_2_power,
            one_minus_beta_2_t=1 - beta_2_t,
            beta_3_t=beta_3_t,
            one_minus_beta_3_t=1 - beta_3_t,
            k=k_t,
            K=K_t
            ))

  # @tf.function  
  def _resource_apply_dense(self, grad, var, apply_state=None) -> None:
    """ Dense implementation of the optimiser apply. Similar to the Adam and
    replaces the unused Sparse function

    Args:
        grad (): Gradient
        var (): Variables
        apply_state ((), optional): function*. Defaults to None.
    """
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    # print(f'Grad Shape: {grad.shape}')
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                           use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = state_ops.assign_add(ref=m, value=m_scaled_g_values,
                    use_locking=self._use_locking, name=None) 

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, 'v')
    v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
    v_t = state_ops.assign(v, v * coefficients['beta_2_t'],
                           use_locking=self._use_locking)
    # print(f'v_scaled_g_values: {v_scaled_g_values}')
    # print(f'v_t-1: {v_t}')
    with ops.control_dependencies([v_t]):
      v_t = state_ops.assign_add(ref=v, value=v_scaled_g_values,
            use_locking=self._use_locking, name=None) 
    # print(f'v_t-2: {v_t}')
    prev_loss = self.prev_loss # self.get_slot(var, 'prev_loss')
    current_loss = self.current_loss # self.get_slot(var, 'current_loss')

    if self.prev_loss is None:
      # prev_loss = state_ops.assign(prev_loss, var,
      #                      use_locking=self._use_locking)
      # W_t = W - lr * m_t / (sqrt(v)+epsilon)
      v_sqrt = math_ops.sqrt(v_t)
      var_update = state_ops.assign_sub(
          var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']),
          use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t])
    else:
      # print(f'_dense:prev_loss:{prev_loss}')
      # print(f'_dense:currrent_loss:{current_loss}')
      # print(f'Var Shape: {var.shape}')
      if (math_ops.abs(current_loss) >= math_ops.abs(prev_loss)):
      # if (tf.cond(tf.greater_equal(current_loss, prev_loss),
      #             lambda: tf.constant(1, tf.int8),
      #             lambda: tf.constant(0, tf.int8))):
        # r = min{(max{k,(L)}),K}
        r_t = math_ops.minimum(
            x=math_ops.maximum(
                x=coefficients['k'],
                y=math_ops.abs(current_loss/prev_loss)
              ),
            y=coefficients['K']
          )
      else:
        # r = min{(max{1/K,(L)}),1/k}
        r_t = math_ops.minimum(
            x=math_ops.maximum(
                x=1/coefficients['K'],
                y=math_ops.abs(current_loss/prev_loss)
              ),
            y=1/coefficients['k']
          )
      # r_t = math_ops.abs(current_loss/prev_loss)
      # d_t = beta3 * d + (1 - beta3) * r
      d = self.get_slot(var, 'd')
      d_scaled_r_values = r_t * coefficients['one_minus_beta_3_t']
      d_t = state_ops.assign(d, d * coefficients['beta_3_t'] + d_scaled_r_values,
                           use_locking=self._use_locking)
      # W_t = W - lr * m_t / (d_t*sqrt(v)+epsilon)
      v_sqrt = math_ops.sqrt(v_t)
      var_update = state_ops.assign_sub(
          var, coefficients['lr'] * m_t / (d_t * v_sqrt + coefficients['epsilon']),
          use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t, d_t])


  def update_loss(self, prev_loss : float32, current_loss : float32,
                  grads_and_vars) -> None:
    """ Custom function added specifically for Robust Adam implementation. TF
    has no meaning to pass loss. This is the only I was able to figure.

    Args:
        prev_loss (float32): Previos Loss value
        current_loss (float32): Currently calculated Loss.
    """
    # print('update_loss')
    if prev_loss is not None:
      # print(f'\n1)PRev_loss and loss: {prev_loss} and {current_loss}')
      self.prev_loss = prev_loss
    else:
      self.prev_loss = 1.0
    self.current_loss = current_loss
    return self.apply_gradients(grads_and_vars, name="FancyMinimize")
  
  def _resource_apply_sparse(self, grad, handle, indices, apply_state):
    """ Unused function in this impleemntation.

    Raises:
        NotImplementedError: Not implemented
    """
    print('_resource_apply_sparse: not implemented')
    raise NotImplementedError

  def get_config(self):
    """ Configuration function used to save and restore model with this optimiser

    Returns:
        config: super.config update from parent class
    """
    # print('get_config')
    config = super().get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay'        : self._serialize_hyperparameter('decay'),
        'beta_1'       : self._serialize_hyperparameter('beta_1'),
        'beta_2'       : self._serialize_hyperparameter('beta_2'),
        'beta_3'       : self._serialize_hyperparameter('beta_3'),
        'k'            : self._serialize_hyperparameter('k'),
        'K'            : self._serialize_hyperparameter('K'),
        'epsilon'      : self.epsilon,
    })
    return config