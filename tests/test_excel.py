#!/usr/bin/python
# %% [markdown]
# Testing Excel reading with Cython
# 
# %%
import os, sys, time
import numpy as np

sys.path.append(os.getcwd() + '/..')
from py_modules.parse_excel import ParseExcelData
from py_modules.utils import diffSoC as py_diffSoC # For testing
from py_modules.utils import Locate_Best_Epoch

from py_modules.parse_excel import Read_Excel_File
from cy_modules.utils import diffSoC as cy_diffSoC # For testing
# %%
if __name__=="__main__":
    columns = ['Current(A)', 'Voltage(V)', 'Temperature (C)_1',
        'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
    ]
    train_dir	  : str ='../Data/A123_Matt_Set'
    r_DST_US_FUDS : range = range(4 , 25)  # Full cycle
# %%
    # Test multicore entire set parser
    tic : float = time.perf_counter()
    tr_ls_df, tr_ls_SoC = ParseExcelData(train_dir, None, columns)
    print(f'Time: {time.perf_counter() - tic}')
# %%
    # Test single file parser
    file = train_dir + '/A1-007-DST-US06-FUDS-20-20120817.xlsx'
    tic : float = time.perf_counter()
    data_df = Read_Excel_File(file, r_DST_US_FUDS, columns)
    print(f'Read Single Excel Py Time: {time.perf_counter() - tic}')

# %%
    # Test the SoC calculator
    samples = 8
    times : np = np.empty(shape=(samples,), dtype=np.float32)
    for i in range(samples):
        tic : float = time.perf_counter()
        SoC = cy_diffSoC(data_df['Charge_Capacity(Ah)'].to_numpy(dtype=np.float32),
                        data_df['Discharge_Capacity(Ah)'].to_numpy(dtype=np.float32))
        times[i] = (time.perf_counter() - tic)
    print(f'SoC with Cy Time over {samples}:\n'
          f'Minumum: {np.min(times)}\n'
          f'Maximum: {np.max(times)}\n\n'
          f'Mean: {np.mean(times)}\n')

    times : np = np.empty(shape=(samples,), dtype=np.float32)
    for i in range(samples):
        tic : float = time.perf_counter()
        SoC = py_diffSoC(data_df['Charge_Capacity(Ah)'].to_numpy(dtype=np.float32),
                        data_df['Discharge_Capacity(Ah)'].to_numpy(dtype=np.float32))
        times[i] = (time.perf_counter() - tic)
    print(f'SoC with Py Time over {samples}:\n'
          f'Minumum: {np.min(times)}\n'
          f'Maximum: {np.max(times)}\n\n'
          f'Mean: {np.mean(times)}\n')
# %%
    tic : float = time.perf_counter()
    epoch : int = Locate_Best_Epoch(
                file_path='../Models/Chemali2017/FUDS-models/history-FUDS.csv'
            )
    print(f"Locating best epoch with Py took: {time.perf_counter() - tic}s")
# %%