#!/usr/bin/python
# %% [markdown]
# Testing Excel reading with Cython
# 
# %%
import os, sys, time

sys.path.append(os.getcwd() + '/..')
from py_modules.parse_excel import ParseExcelData
from py_modules.utils import diffSoC # For testing

# %%
if __name__=="__main__":
    columns = ['Current(A)', 'Voltage(V)', 'Temperature (C)_1',
        'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
    ]
    train_dir	  : str ='../Data/A123_Matt_Set'
    r_DST_US_FUDS : range = range(4 , 25)  # Full cycle
    tic : float = time.perf_counter()
    tr_ls_df, tr_ls_SoC = ParseExcelData(train_dir, None, columns)
    print(f'Time: {time.perf_counter() - tic}')
# %%