# %%
from julia.api import Julia
import numpy as np
#julia.install()

j = Julia(compiled_modules=False)
fn = j.include('plotting.jl')
fn
x = np.array([[1,2,3], [4,5,6]], dtype=np.float64)
fn(x)

# %%
from julia import Main
Main.include("julscr1.jl")