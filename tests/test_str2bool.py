# %%
import time, sys, os
import numpy as np

sys.path.append(os.getcwd() + '/..')
from py_modules.utils import str2bool as py_str2bool

from c_modules.utils import str2bool as c_str2bool

from cy_modules.utils import str2bool as cy_str2bool

test_bools = [
        "yes", "true", "True", "False", "false", "y", "t", "1", "0"
    ]
test_results =  [
         True, True, True, False, False, True, True, True, False
    ]

# %%
if __name__ == "__main__":
    # List Python
    timers_py = []
    timers_c = []
    timers_cy = []
    for i in range(len(test_bools)):
        # Python version
        start = time.perf_counter()
        py_str2bool(test_bools[i])
        timers_py.append(time.perf_counter() - start)
        if(py_str2bool(test_bools[i]) != test_results[i]):
            print(f"Py at {i} failed")
        
        # # C based version
        start = time.perf_counter()
        c_str2bool(test_bools[i])
        timers_c.append(time.perf_counter() - start)
        if(c_str2bool(test_bools[i]) != test_results[i]):
            print(f"C at {i} failed")

        # Cython
        start = time.perf_counter()
        cy_str2bool(test_bools[i])
        timers_cy.append(time.perf_counter() - start)
        if(cy_str2bool(test_bools[i]) != test_results[i]):
            print(f"Cy at {i} failed")
    
    mult = 1000000
    print('\nPython: %.6fns' % (np.mean(timers_py)*mult) )
    print('C:      %.6fns' % (np.mean(timers_c)*mult) )
    print('Cython: %.6fns' % (np.mean(timers_cy)*mult) )

# %%
