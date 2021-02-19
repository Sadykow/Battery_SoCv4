#!/usr/bin/python
# %% [markdown]
# Data received by Sam from UQ ... at 9 of January 2020.
#
#
# %%
import pandas as pd             # File read
import matplotlib as mpl        # Plot functionality
import matplotlib.pyplot as plt

import numpy as np
#import tensorflow.experimental.numpy as tnp 
# Define plot sizes
mpl.rcParams['figure.figsize'] = (18, 16)
mpl.rcParams['axes.grid'] = False

# %%
df : pd.DataFrame = pd.read_csv(
                #filepath_or_buffer='Data/UQ_Endurance/Copy_of_20201210-0055704.csv',
                filepath_or_buffer='UQ_Endurance/uq_data.csv',
                sep=',', delimiter=None,
                header='infer', names=None, index_col=None,
                usecols=None, squeeze=False, prefix=None,
                mangle_dupe_cols=True, dtype=None, engine=None,
                converters=None, true_values=None, false_values=None,
                skipinitialspace=False, skiprows=None, skipfooter=0,
                nrows=None, na_values=None, keep_default_na=True,
                na_filter=True, verbose=False, skip_blank_lines=True,
                parse_dates=False, infer_datetime_format=False,
                keep_date_col=False, date_parser=None, dayfirst=False,
                cache_dates=True, iterator=False, chunksize=None,
                compression='infer', thousands=None, decimal='.',
                lineterminator=None, quotechar='"', quoting=0,
                doublequote=True, escapechar=None, comment=None,
                encoding=None, dialect=None, error_bad_lines=True,
                warn_bad_lines=True, delim_whitespace=False,
                low_memory=True, memory_map=False, float_precision=None,
                #storage_options=None
                )
df.plot(subplots=True)
# %%
# Breaking up on multiple plots
print(df.columns)
df[df['BMS Pack Voltage'] == 0] = None
df.dropna(inplace=True)
df = df.reset_index(drop=True)
# %%
# Plot
START = 0
END = df.shape[0]
SPLIT = int(END/2)-380
STEP = 1
plt.figure()
plt.plot(df['Time'][START:SPLIT:], df['WS Current'][START:SPLIT:])
plt.xlabel('Time (s)')
plt.ylabel('Current (A) - Not specified')
plt.title('Current - Part 1')
plt.grid()
plt.figure()
plt.plot(df['Time'][START:SPLIT:], df['BMS Pack Voltage'][START:SPLIT:], color='r')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V) - Not specified')
plt.title('Voltage - Part 1')
plt.grid()
plt.figure()import numpy as np
plt.plot(df['Time'][START:SPLIT:], df['MB1 High Temp'][START:SPLIT:], label='MB1')
plt.plot(df['Time'][START:SPLIT:], df['MB2 High Temp'][START:SPLIT:], label='MB2')
plt.plot(df['Time'][START:SPLIT:], df['MB3 High Temp'][START:SPLIT:], label='MB3')
plt.plot(df['Time'][START:SPLIT:], df['MB4 High Temp'][START:SPLIT:], label='MB4')
plt.plot(df['Time'][START:SPLIT:], df['MB5 High Temp'][START:SPLIT:], label='MB5')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (C)')
plt.title('Temperature - Part 1')
plt.legend()
plt.grid()

# %%
START = SPLIT+1000
END = df.shape[0]-100
SPLIT = END
# %%
plt.figure()
plt.plot(df['Time'][START:SPLIT:], df['WS Current'][START:SPLIT:])
plt.xlabel('Time (s)')
plt.ylabel('Current (A) - Not specified')
plt.title('Current - Part 2')
plt.grid()
plt.figure()
plt.plot(df['Time'][START:SPLIT:], df['BMS Pack Voltage'][START:SPLIT:], color='r')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V) - Not specified')
plt.title('Voltage - Part 2')
plt.grid()
plt.figure()
plt.plot(df['Time'][START:SPLIT:], df['MB1 High Temp'][START:SPLIT:], label='MB1')
plt.plot(df['Time'][START:SPLIT:], df['MB2 High Temp'][START:SPLIT:], label='MB2')
plt.plot(df['Time'][START:SPLIT:], df['MB3 High Temp'][START:SPLIT:], label='MB3')
plt.plot(df['Time'][START:SPLIT:], df['MB4 High Temp'][START:SPLIT:], label='MB4')
plt.plot(df['Time'][START:SPLIT:], df['MB5 High Temp'][START:SPLIT:], label='MB5')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (C)')
plt.title('Temperature - Part 2')
plt.legend()
plt.grid()
# %%
# Cell type is LG HG2
# 18650
# Apparently 18s 7p
# 5 in Series

# 345 Volts - 5 in series - 69V each
# 69V - 18 in series - 3.83V each
# 60A - 7 in paralel - 8.6A each cell

# Nominal Capacity: 3000mAh
# Nominal voltage: 3.50V
# Standard charge: 1500mA, 4.2V, 50mA
# Max. charge voltage: 4.20V+/-0.05V
# Max. charge current: 4000mA
# Standard discharge: 600mA down to 2.5V
# Fast discharge: 10000mA, 20000mA down to 2.5V
# Max. continuous discharge: 20000mA
# Cycle life: 300 at 10A, 200 at 20A both with 4A charge, ramaning capacity minimum 70%
# Weight: 47.0g
# Operating temperature: charge 0°C ~ 50°C, discharge: -20°C ~ 75°C
# Storage temperature: 1 month: -20°C ~60°C, 3 month: -20°C ~ 45°C, 1 year: -20°C ~ 20°C
# 
# %%
# Calculation for QUT HV batteries
df['Power'] = df['WS Current']*df['BMS Pack Voltage']
df['C_U'] = df['Power'] / 455 
df['C_L'] = df['Power'] / 488

# I^2*R
AC_R : float = 0.007
DC_R : float = 0.016

# DC
df['DC_U_PowerLoss'] = np.power(df['C_U'],
                                2
                            ) * DC_R
df['DC_L_PowerLoss'] = np.power(df['C_L'],
                                2
                            ) * DC_R
# AC
df['AC_U_PowerLoss'] = np.power(df['C_U'],
                                2
                            ) * AC_R
df['AC_L_PowerLoss'] = np.power(df['C_L'],
                                2
                            ) * AC_R

# Power Loss
print('Power Loss to Heat RMS Upper DC: {value}'.format(
    value=np.sqrt(np.mean(
                    a=np.power(
                        df['C_U'],
                        2),
                    axis=None)
                )*(140*DC_R)
))
print('Power Loss to Heat RMS Lower DC: {value}'.format(
    value=np.sqrt(np.mean(
                    a=np.power(
                        df['C_L'],
                        2),
                    axis=None)
                )*(140*DC_R)
))
#plt.plot(df['Power'])