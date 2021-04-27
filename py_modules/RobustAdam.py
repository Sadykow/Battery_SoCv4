from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.keras import backend
#from tensorflow.python.util.tf_export import keras_export
import tensorflow as tf
from numpy import float32
#@keras_export('tf.keras.optimizers.RobustAdam')
class RobustAdam(tf.keras.optimizers.Optimizer):
  
  epsilon : tf.float32 = 10e-8
  _is_first : bool = True

  # Robust part
  prev_loss : tf.Variable = None
  current_loss : tf.Variable = None

  i_init : int = 0
  t : int = 0

  mae_loss = tf.losses.MeanAbsoluteError()

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
  
  def mse_loss(self, y_true, y_pred):
    """Computes the mean squared error between labels and predictions.

    After computing the squared distance between the inputs, the mean value over
    the last dimension is returned.

    `loss = mean(square(y_true - y_pred), axis=-1)`

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(), np.mean(np.square(y_true - y_pred), axis=-1))

    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

    Returns:
      Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return backend.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)[0]

  def rmse_loss(self, y_true, y_pred):
    """Computes the mean squared error between labels and predictions.
    #? Root Mean Squared Error loss function
    After computing the squared distance between the inputs, the mean value over
    the last dimension is returned.

    `loss = mean(square(y_true - y_pred), axis=-1)`

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(), np.mean(np.square(y_true - y_pred), axis=-1))

    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

    Returns:
      Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return backend.sqrt(
        x=backend.mean(
            x=math_ops.squared_difference(x=y_pred, y=y_true),
            axis=-1,
            keepdims=False
          )
      )[0]
  
  def _create_slots(self, var_list : list[tf.Variable]):
    """ For each model variable, create the optimizer variable associated
    with it. TensorFlow calls these optimizer variables "slots".
    For momentum optimization, we need one momentum slot per model variable.

    Create slots for the first, second and third moments.
    Separate for-loops to respect the ordering of slot variables from v1.

    Args:
        var_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    for var in var_list:
      self.add_slot(var, 'm')    
    for var in var_list:
      self.add_slot(var, 'v')
    for var in var_list:
      self.add_slot(var, 'd')
    for var in var_list:
      self.add_slot(var, 'r')
    for var in var_list:
      self.add_slot(var, 'prev_var')
    
  def _prepare_local(self, var_device, var_dtype, apply_state):
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
  
  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
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
    with ops.control_dependencies([v_t]):
      v_t = state_ops.assign_add(ref=v, value=v_scaled_g_values,
            use_locking=self._use_locking, name=None) 
    prev_var = self.get_slot(var, 'prev_var')

    #if prev_var is None:
    if self.prev_loss is None:
      prev_var = state_ops.assign(prev_var, var,
                           use_locking=self._use_locking)
      # W_t = W - lr * m_t / (sqrt(v)+epsilon)
      v_sqrt = math_ops.sqrt(v_t)
      var_update = state_ops.assign_sub(
          var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']),
          use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t, prev_var])
    else:
      r = self.get_slot(var, 'r')
      if math_ops.abs(self.current_loss) >= math_ops.abs(self.prev_loss):
        # r = min{(max{k,(L)}),K}
        r_loss = math_ops.minimum(
            x=math_ops.maximum(
                x=coefficients['k'],
                y=math_ops.abs(self.current_loss/self.prev_loss)
              ),
            y=coefficients['K']
          )
      else:
        # r = min{(max{1/K,(L)}),1/k}
        r_loss = math_ops.minimum(
            x=math_ops.maximum(
                x=1/coefficients['K'],
                y=math_ops.abs(self.current_loss/self.prev_loss)
              ),
            y=1/coefficients['k']
          )
      ones = tf.ones(
          shape=r.shape, dtype=tf.dtypes.float32, name=None
        )
      r_t = state_ops.assign(r, ones * r_loss,
                           use_locking=self._use_locking)
      
      # d_t = beta3 * d + (1 - beta3) * r      
      d = self.get_slot(var, 'd')
      d_scaled_r_values = r_t * coefficients['one_minus_beta_3_t']
      d_t = state_ops.assign(d, d * coefficients['beta_3_t'],
                           use_locking=self._use_locking)
      with ops.control_dependencies([d_t]):
        #! I blody repmat() it manualy if I have to 4 it!
        d_t = state_ops.assign_add(ref=d, value=4.0*d_scaled_r_values,
                    use_locking=self._use_locking, name=None)
      # print(f'\nR_t: {r_t}\nd: {d}\n d_t: {d_t}\n')

      prev_var = state_ops.assign(prev_var, var,
                           use_locking=self._use_locking)
      # W_t = W - lr * m_t / (d_t*sqrt(v)+epsilon)
      v_sqrt = math_ops.sqrt(v_t)
      var_update = state_ops.assign_sub(
          var, coefficients['lr'] * m_t / (d_t * v_sqrt + coefficients['epsilon']),
          use_locking=self._use_locking)
      # if (d_t.shape[0] == 1):
      #   normal = v_sqrt + coefficients['epsilon']
      #   roadam = d_t * v_sqrt + coefficients['epsilon']
      #   print(f'Normal: {normal}')
      #   print(f'RoAdam: {roadam}')
      return control_flow_ops.group(*[var_update, m_t, v_t, prev_var, d_t])


  def update_loss(self, prev_loss : tf.float32, current_loss : tf.float32):
    if prev_loss is not None:
      # print(f'\n1)PRev_loss and loss: {prev_loss} and {current_loss}')
      self.prev_loss = prev_loss
      self.current_loss = current_loss
    # else:
    #   print('\nNo Prev Value')
  
  def _resource_apply_sparse(self, grad, handle, indices, apply_state):
    raise NotImplementedError

  def get_config(self):
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