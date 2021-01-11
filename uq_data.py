#!/usr/bin/python
# %% [markdown]
# Data received by Sam from UQ ... at 9 of January 2020.
#
#
# %%
import pandas as pd             # File read
import matplotlib as mpl        # Plot functionality
import matplotlib.pyplot as plt

import tensorflow.experimental.numpy as tnp 
# Define plot sizes
mpl.rcParams['figure.figsize'] = (18, 16)
mpl.rcParams['axes.grid'] = False

# %%
df : pd.DataFrame = pd.read_csv(filepath_or_buffer='Data/UQ_Endurance/Copy_of_20201210-0055704.csv',
                sep=',', delimiter=None,
                header='infer', names=None, index_col=None,
                usecols=None, squeeze=False, prefix=None,
                mangle_dupe_cols=True, dtype=tnp.float32, engine=None,
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
plt.figure()
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
Cell type is LG HG2
18650
Apparently 18s 7p
5 in Series
