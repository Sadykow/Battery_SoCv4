#from tensorflow.python.ops import state_ops
#from tensorflow.python.util.tf_export import keras_export
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
#@keras_export('tf.keras.optimizers.RobustAdam')
class RobustAdam(tf.keras.optimizers.Optimizer):
  
  epsilon : tf.float32 = 10e-8
  _is_first : bool = True

  # Robust part
  prediction : tf.Variable = 0.0
  actual : tf.Variable = 0.0
  prev_loss : tf.Variable = 0.0

  i_init : int = 0
  t : int = 0

  loss = lambda var : tf.sqrt(tf.reduce_mean(tf.square(var), axis=0))
      
  def __init__(self, name : str ='RobustAdam',
                lr_rate: tf.float32 = 0.001,
                beta_1 : tf.float32 = 0.9,
                beta_2 : tf.float32 = 0.999,
                beta_3 : tf.float32 = 0.999,
                epsilon :tf.float32 = 1e-7,
                _is_first : bool = False,
                **kwargs):
    super(RobustAdam, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", lr_rate))
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self._set_hyper('beta_3', beta_3)
    self.epsilon = epsilon # or backend_config.epsilon()
    self._is_first = _is_first
              
  
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
      self.add_slot(var, 'm_t')
    for var in var_list:
      self.add_slot(var, 'v_t')
    for var in var_list:
      self.add_slot(var, 'd_t')
    for var in var_list:
      self.add_slot(var, 'r_t')
    
    # # Adding previos variables
    for var in var_list:
      self.add_slot(var, 'prev_var_0')
    for var in var_list:
      self.add_slot(var, 'prev_var_1')
    for var in var_list:
      self.add_slot(var, 'prev_var_2')
    for var in var_list:
      self.add_slot(var, 'prev_var_3')
    for var in var_list:
      self.add_slot(var, 'prev_var_4')
    for var in var_list:
      self.add_slot(var, 'var_update')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(RobustAdam, self)._prepare_local(var_device, var_dtype, apply_state)
    local_step = tf.cast(self.iterations + 1, var_dtype)
    beta_1_t = self._get_hyper('beta_1', var_dtype)
    beta_2_t = self._get_hyper('beta_2', var_dtype)
    beta_3_t = self._get_hyper('beta_3', var_dtype)
    beta_1_power = tf.pow(beta_1_t, local_step)
    beta_2_power = tf.pow(beta_2_t, local_step)
    lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
         (tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
    apply_state[(var_device, var_dtype)].update(
        dict(
            lr=lr,                
            epsilon=tnp.float32(self.epsilon),
            beta_1_t=beta_1_t,
            beta_1_power=beta_1_power,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            beta_2_power=beta_2_power,
            one_minus_beta_2_t=1 - beta_2_t,
            beta_3_t=beta_3_t,
            one_minus_beta_3_t=1 - beta_3_t
            ))

  @tf.function
  def _resource_apply_dense(self, grad, var, apply_state=None):
    """ Update the slots and perform one optimization step for one model
    variable.

    Args:
        grad ([type]): [description]
        var ([type]): [description]
        apply_state ([type]): [description]

    Returns:
        [type]: [description]
    """
    var_device, var_dtype = var.device, var.dtype.base_dtype        
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                or self._fallback_apply_state(var_device, var_dtype))
    m = self.get_slot(var=var, slot_name='m')
    v = self.get_slot(var=var, slot_name='v')
    d = self.get_slot(var=var, slot_name='d')

    m_t = self.get_slot(var=var, slot_name='m_t')
    v_t = self.get_slot(var=var, slot_name='v_t')
    d_t = self.get_slot(var=var, slot_name='d_t')
    r_t = self.get_slot(var=var, slot_name='r_t')
    
    prev_var_0 = self.get_slot(var=var, slot_name='prev_var_0')
    prev_var_1 = self.get_slot(var=var, slot_name='prev_var_1')
    prev_var_2 = self.get_slot(var=var, slot_name='prev_var_2')
    prev_var_3 = self.get_slot(var=var, slot_name='prev_var_3')
    prev_var_4 = self.get_slot(var=var, slot_name='prev_var_4')

    var_update = self.get_slot(var=var, slot_name='var_update')

    # m_t = beta1 * m + (1 - beta1) * g_t
    m_scaled_g_values = coefficients['one_minus_beta_1_t'] * grad
    
    # m_t = state_ops.assign(ref=m, value=m * coefficients['beta_1_t'],
    #                 use_locking=self._use_locking,
    #                 validate_shape=None, name='m_t')
    # with tf.control_dependencies([m_t]):        
    #     m_t = state_ops.assign_add(ref=m_t, value=m_scaled_g_values,
    #                 use_locking=self._use_locking, name=None)            
    m_t.assign(m * coefficients['beta_1_t'])
    m_t.assign_add(m_scaled_g_values)
    
    # m_hat = m_t / (1-beta_1**t)
    # m_hat = state_ops.assign(ref=m_t, value=m_t / (1-(coefficients['beta_1_t']**self.t)),
    #                 use_locking=self._use_locking,
    #                 validate_shape=None, name='m_hat')

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v_scaled_g_values = coefficients['one_minus_beta_2_t']*(grad * grad)
    # v_t = state_ops.assign(ref=v, value=v * coefficients['beta_2_t'],
    #                 use_locking=self._use_locking,
    #                 validate_shape=None, name='v_t')
    # with tf.control_dependencies([v_t]):
    #     v_t = state_ops.assign_add(ref=v_t, value=v_scaled_g_values,
    #                 use_locking=self._use_locking, name=None)        
    v_t.assign(v * coefficients['beta_2_t'])
    v_t.assign_add(v_scaled_g_values)
    
    # m_hat = m_t / (1-beta_1**t)
    # v_hat = state_ops.assign(ref=v_t, value=v_t / (1-(coefficients['beta_2_t']**self.t)),
    #                 use_locking=self._use_locking,
    #                 validate_shape=None, name='m_hat')
                    
    if self._is_first:
      # var_update = state_ops.assign_sub(
      #     ref=var, 
      #     value=coefficients['lr'] * m_t /\
      #         (tf.sqrt(v_t) + coefficients['epsilon']),
      #     use_locking=self._use_locking)
      var_update.assign_sub(coefficients['lr'] * m_t /\
              (tf.sqrt(v_t) + coefficients['epsilon']))
      
      if self.i_init == 0:
        self.prev_var_0 = var_update
        # prev_var_0.assign(var_update)
      elif self.i_init == 1:
        self.prev_var_1 = var_update
        # prev_var_1.assign(var_update)
      elif self.i_init == 2:
        self.prev_var_2 = var_update
        # prev_var_2.assign(var_update)
      elif self.i_init == 3:
        self.prev_var_3 = var_update
        # prev_var_3.assign(var_update)                
      else:
        self.prev_var_4 = var_update
        # prev_var_4.assign(var_update)
          
      self.i_init += 1
      
      return [var_update, m_t, v_t]
    else :
      # r_t = ||L(var)/L(prev_var)||
      if self.i_init % 5 == 0:
        # r_t = tf.abs(x=tf.divide(
        #         x=var,
        #         y=self.prev_var_0,
        #         name='Robust'
        #     )
        # )
        r_t.assign((var)/(prev_var_0))
        prev_var_0.assign(var)

        # r_t = state_ops.assign(
        #         ref=var,
        #         value=var/prev_var_0,
        #         use_locking=self._use_locking,
        #         validate_shape=None, name='r_t'
        #     )
        # prev_var_0 = var
      elif self.i_init % 5 == 1:
        # r_t = state_ops.assign(
        #         ref=var,
        #         value=var/prev_var_1,
        #         use_locking=self._use_locking,
        #         validate_shape=None, name='r_t'
        #     )
        r_t.assign((var)/(prev_var_1))
        prev_var_1.assign(var)
        # r_t = tf.abs(x=tf.divide(
        #         x=var,
        #         y=prev_var_1,
        #         name='Robust'
        #     )
        # )
        # prev_var_1 = var
      elif self.i_init  % 5 == 2:
        # r_t = state_ops.assign(
        #         ref=var,
        #         value=var/prev_var_2,
        #         use_locking=self._use_locking,
        #         validate_shape=None, name='r_t'
        #     )
        
        r_t.assign((var)/(prev_var_2))
        prev_var_2.assign(var)
        # r_t = tf.abs(x=tf.divide(
        #         x=var,
        #         y=prev_var_2,
        #         name='Robust'
        #     )
        # )
        # self.prev_var_2 = var                
      elif self.i_init  % 5 == 3:
        # r_t = state_ops.assign(
        #         ref=var,
        #         value=var/prev_var_3,
        #         use_locking=self._use_locking,
        #         validate_shape=None, name='r_t'
        #     )
        
        r_t.assign((var)/(prev_var_3))
        prev_var_3.assign(var)
        # r_t = tf.abs(x=tf.divide(
        #         x=var,
        #         y=prev_var_3,
        #         name='Robust'
        #     )
        # )
        # prev_var_3 = var
      else:
        # r_t = state_ops.assign(
        #         ref=var,
        #         value=var/prev_var_4,
        #         use_locking=self._use_locking,
        #         validate_shape=None, name='r_t'
        #     )
        
        r_t.assign((var)/(prev_var_4))
        prev_var_4.assign(var)
        # r_t = tf.abs(x=tf.divide(
        #         x=var,
        #         y=prev_var_4,
        #         name='Robust'
        #     )
        # )
        # prev_var_4 = var                
      self.i_init += 1
      #r_t = current_loss / self.prev_loss
      #self.prev_loss = current_loss
      
      # d_t = beta3 * d + (1 - beta3) * r_t
      d_scaled_L = coefficients['one_minus_beta_3_t']*(r_t)
      # d_t = state_ops.assign(ref=d, value=d*coefficients['beta_3_t'],
      #                 use_locking=self._use_locking,
      #                 validate_shape=None, name='d_t')
      
      # with tf.control_dependencies([d_t]):
      #     d_t = state_ops.assign_add(ref=d_t, value=d_scaled_L,
      #                 use_locking=self._use_locking, name=None)
      d_t.assign(d*coefficients['beta_3_t'])
      d_t.assign_add(d_scaled_L)
      
      # var_update = state_ops.assign_sub(
      #     ref=var, 
      #     value=coefficients['lr'] * m_t /\
      #           (d_t*tf.sqrt(v_t) + coefficients['epsilon']),
      #     use_locking=self._use_locking)
      var_update.assign_sub(coefficients['lr'] * m_t /\
                (d_t*tf.sqrt(v_t) + coefficients['epsilon']))
      return [var_update, m_t, v_t]

  def update_labels(self, loss_value : tf.float32, label : tf.float32,
                    first_run : bool = False, t : int = 0):
    self.prediction = tf.Variable(loss_value, dtype=tf.float32)
    self.actual = tf.Variable(label, dtype=tf.float32)
    self.first_run = first_run
    self.t = t

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
        'epsilon'      : self.epsilon,
    })
    return config