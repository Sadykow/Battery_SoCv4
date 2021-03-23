import tensorflow as tf

class SGOptimizer(tf.keras.optimizers.Optimizer):
  """Gradient Descent
    New_weight = weight  - eta * rate of change of error wrt weight
    w = w - η* ∂E/∂w 

    Taken from following blog:
    https://cloudxlab.com/blog/writing-custom-optimizer-in-tensorflow-and-keras/
    
    Complete code:
    https://github.com/cloudxlab/ml/blob/master/exp/Optimizer_2.ipynb

  Args:
      tf ([type]): [description]
  """
  def __init__(self, learning_rate=0.01, name="SGOptimizer", **kwargs):
    """Call super().__init__() and use _set_hyper() to store hyperparameters"""
    super().__init__(name, **kwargs)
    # handle lr=learning_rate
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._is_first = True
    
  def _create_slots(self, var_list):
    """For each model variable, create the optimizer variable associated with
    it. TensorFlow calls these optimizer variables "slots".
    For momentum optimization, we need one momentum slot per model variable.
    """
    for var in var_list:
        self.add_slot(var, "pv") #previous variable i.e. weight or bias
    for var in var_list:
        self.add_slot(var, "pg") #previous gradient


  @tf.function
  def _resource_apply_dense(self, grad, var):
    """Update the slots and perform one optimization step for one model variable
    """
    var_dtype = var.dtype.base_dtype
    lr_t = self._decayed_lr(var_dtype) # handle learning rate decay
    new_var_m = var - grad * lr_t
    pv_var = self.get_slot(var, "pv")
    pg_var = self.get_slot(var, "pg")
    
    if self._is_first:
        print("First Pass")
        self._is_first = False
        new_var = new_var_m
        tf.print(new_var)
        print("=========\n\n")
    else:
        print("Second Pass")
        cond = grad*pg_var >= 0
        #tf.print(cond)
        avg_weights = (pv_var + var)/2.0
        new_var = tf.where(cond, new_var_m, avg_weights)
        print("=========\n\n")
    pv_var.assign(var)
    pg_var.assign(grad)
    var.assign(new_var)

  def _resource_apply_sparse(self, grad, var):
    raise NotImplementedError
  
  def get_config(self):
    base_config = super().get_config()
    return {
        **base_config,
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
    }
