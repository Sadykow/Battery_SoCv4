#!/usr/bin/python
# %% [markdown]
# Ploting windowing for JA-Analysis
# %%
import os
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib as mpl        # Plot functionality
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.backend import shape

# Define plot sizes
mpl.rcParams['figure.figsize'] = (16, 6)
mpl.rcParams['axes.grid'] = False

train_dir : str = 'Data/A123_Matt_Set'
valid_dir : str = 'Data/A123_Matt_Val'
columns   : list[str] = [
                        'Test_Time(s)',
                        'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                        'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                    ]

def Read_Excel_File(path : str, profile : range,
                    columns : list[str]) -> pd.DataFrame:
    """ Reads Excel File with all parameters. Sheet Name universal, columns,
    type taken from global variables initialization.

    Args:
        path (str): Path to files with os.walk

    Returns:
        pd.DataFrame: Single File frame.
    """
    try:
      df : pd.DataFrame = pd.read_excel(io=path,
                        sheet_name='Channel_1-006',
                        header=0, names=None, index_col=None,
                        usecols=['Step_Index'] + columns,
                        squeeze=False,
                        dtype=np.float32,
                        engine='openpyxl', converters=None, true_values=None,
                        false_values=None, skiprows=None, nrows=None,
                        na_values=None, keep_default_na=True, na_filter=True,
                        verbose=False, parse_dates=False, date_parser=None,
                        thousands=None, comment=None, skipfooter=0,
                        convert_float=True, mangle_dupe_cols=True
                      )
    except:
      df : pd.DataFrame = pd.read_excel(io=path,
                        sheet_name='Channel_1-005',
                        header=0, names=None, index_col=None,
                        usecols=['Step_Index'] + columns,
                        squeeze=False,
                        dtype=np.float32,
                        engine='openpyxl', converters=None, true_values=None,
                        false_values=None, skiprows=None, nrows=None,
                        na_values=None, keep_default_na=True, na_filter=True,
                        verbose=False, parse_dates=False, date_parser=None,
                        thousands=None, comment=None, skipfooter=0,
                        convert_float=True, mangle_dupe_cols=True
                      )
    df = df[df['Step_Index'].isin(profile)]
    df = df.reset_index(drop=True)
    df = df.drop(columns=['Step_Index'])
    df = df[columns]   # Order columns in the proper sequence
    return df

def diffSoC(chargeData   : pd.Series,
            discargeData : pd.Series) -> pd.Series:
    """ Return SoC based on differnece of Charge and Discharge Data.
    Data in range of 0 to 1.
    Args:
        chargeData (pd.Series): Charge Data Series
        discargeData (pd.Series): Discharge Data Series

    Raises:
        ValueError: If any of data has negative
        ValueError: If the data trend is negative. (end-beg)<0.

    Returns:
        pd.Series: Ceil data with 2 decimal places only.
    """
    # Raise error
    if((any(chargeData) < 0)
        |(any(discargeData) < 0)):
        raise ValueError("Parser: Charge/Discharge data contains negative.")
    return np.round((chargeData - discargeData)*100)/100

def roundup(x : float, factor : int = 10) -> int:
    """ Round up to a factor. Uses it to create hidden neurons, or Buffer size.
    Args:
        x (float): Original float value.
        factor (float): Factor towards which it has to be rounder

    Returns:
        int: Rounded up value based on factor.
    """
    if(factor == 10):
        return int(np.ceil(x / 10)) * 10
    elif(factor == 100):
        return int(np.ceil(x / 100)) * 100
    elif(factor == 1000):
        return int(np.ceil(x / 1000)) * 1000
    else:
        print("Factor of {} not implemented.".format(factor))
        return None

def create_Batch_dataset(X : list[np.ndarray], Y : list[np.ndarray],
                    look_back : int = 1
                    ) -> tuple[np.ndarray, np.ndarray]:
    
    batch : int = len(X)
    dataX : list[np.ndarray] = []
    dataY : list[np.ndarray] = []
    
    for i in range(0, batch):
        d_len : int = X[i].shape[0]-look_back
        dataX.append(np.zeros(shape=(d_len, look_back, X[i].shape[1]),
                    dtype=np.float32))
        dataY.append(np.zeros(shape=(d_len,), dtype=np.float32))    
        for j in range(0, d_len):
            #dataX[i, j, :, :] = dataset[i:(i+look_back), j:j+1]
            #dataY[i, j]       = dataset[i + look_back, j:j+1]
            dataX[i][j,:,:] = X[i][j:(j+look_back), :]  
            dataY[i][j]     = Y[i][j+look_back,]
    return dataX, dataY

def smooth(y, box_pts: int) -> np.array:
    """ Smoothing data using numpy convolve. Based on the size of the
    averaging box, data gets smoothed.
    Here it used in following form:
    y = V/(maxV-minV)
    box_pts = 500

    Args:
        y (pd.Series): A data which requires to be soothed.
        box_pts (int): Number of points to move averaging box

    Returns:
        np.array: Smoothed data array
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
    
#? Getting training data and separated file by batch
for _, _, files in os.walk(train_dir):
    files.sort(key=lambda f: int(f[-13:-5])) # Sort by last dates
    # Initialize empty structures
    train_X : list[pd.DataFrame] = []
    train_Y : list[pd.DataFrame] = []
    for file in files[0:1]:
        X : pd.DataFrame = Read_Excel_File(train_dir + '/' + file,
                                    range(22,25), columns) #! or 21
                                    #range(4,12), columns)
        Y : pd.DataFrame = pd.DataFrame(
                data={'SoC' : diffSoC(
                            chargeData=X.loc[:,'Charge_Capacity(Ah)'],
                            discargeData=X.loc[:,'Discharge_Capacity(Ah)']
                            )},
                dtype=np.float32
            )
        X = X[['Test_Time(s)','Current(A)','Voltage(V)','Temperature (C)_1']]
        train_X.append(X)
        train_Y.append(Y)
scaler_MM : MinMaxScaler    = MinMaxScaler(feature_range=(0, 1))
sample_size : int = 0
for i in range(0, len(train_X)):
    #! Scale better with STD on voltage
    #train_X[i].iloc[:,0] = scaler_CC.fit_transform(np.expand_dims(train_X[i]['Current(A)'], axis=1))
    #train_X[i].iloc[:,1] = scaler_VV.fit_transform(np.expand_dims(train_X[i]['Voltage(V)'], axis=1))    
    train_Y[i] = scaler_MM.fit_transform(train_Y[i])
    #train_X[i] = train_X[i].to_numpy()
    sample_size += train_X[i].shape[0]
train_X[0]['Test_Time(s)'] = train_X[0]['Test_Time(s)']-train_X[0]['Test_Time(s)'][0]
time_minutes = np.linspace(0,int(train_X[0]['Test_Time(s)'].iloc[-1]/60)+1,train_X[0]['Test_Time(s)'].shape[0])
train_X[0]['Test_Time(m)'] = time_minutes
# %% [markdown]
# # Performing Plots generation from 10C example
# %%
# train_X[0]['V_smooth(V)'] = smooth(train_X[0]['Voltage(V)'],600)
# sns.set(rc={'figure.figsize':(16,8)})
# plt.figure(figsize=(24, 8))

V = sns.relplot(x='Test_Time(m)', y='Voltage(V)', kind="line",
                data=train_X[0][:-200], linewidth=1, color='r', legend=False)
V.fig.set_size_inches(12, 8)
# plt.xlim([0,120])
plt.ylim(2.25, 3.78)
# plt.title('Single Cycle - Voltage - 3 hours', size=16)
plt.ylabel('(V)', size=16, rotation=0)
plt.xlabel('Time (minutes)', size=16)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(which='major', alpha=1)

width : float = 1.5
# Box 1
x1,x2,y1,y2  = 0, 10, 2.6, 3.75
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box 2
x1,x2,y1,y2  = 2.50, 12.50, 2.55, 3.70
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box 3
x1,x2,y1,y2  = 6.00, 16.00, 2.53, 3.68
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box Last
x1,x2,y1,y2  = 115.00, 125.00, 2.55, 3.4
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Middle dots
plt.plot([40.00,60.00,80.00], [3.62,3.6,3.58], '.k', linewidth=width)
plt.plot([40.00,60.00,80.00], [2.72,2.7,2.68], '.k', linewidth=width)
#g.fig.autofmt_xdate()
# sns.boxplot(x='Test_Time(s)', y='Voltage(V)', data=train_X[0], 
#                  showcaps=False,boxprops={'facecolor':'None', "zorder":10},
#                  showfliers=False,whiskerprops={'linewidth':0, "zorder":10},
#                  ax=V, zorder=10)

V.fig.savefig('../windowingPlots2/1-Voltage-ext.svg', transparent=True,
                bbox_inches = "tight")
# V.fig.savefig('/mnt/WORK/work/MPhil(CCS)/ThesisDefense/Batteries/1-Voltage.svg', transparent=True,
#                 bbox_inches = "tight")
# %%
# train_X[0]['C_smooth(A)'] = smooth(train_X[0]['Current(A)'],20)
I = sns.relplot(x='Test_Time(m)', y='Current(A)', kind="line",
                data=train_X[0][:-200], linewidth=1.0, color='b', legend=False)
I.fig.set_size_inches(12, 8)
#plt.xlim(-100, 40000)
#plt.ylim(2.25, 3.75)
#g.fig.autofmt_xdate()
# plt.title('Single Cycle - Current - 3 hours', size=16)
plt.ylabel('(A)', size=16, rotation=0)
plt.xlabel('Time (minutes)', size=16)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(which='major', alpha=1)

width : float = 1.5
# Box 1
x1,x2,y1,y2  = 0, 10, -4, 2
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box 2
x1,x2,y1,y2  = 2.50, 12.50, -3.9, 2.1
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box 3
x1,x2,y1,y2  = 6.00, 16.00, -4.1, 1.9
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box Last
x1,x2,y1,y2  = 115.0, 125.0, -4, 2
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Middle dots
plt.plot([40.00,60.00,80.00], [ 2, 2, 2], '.k', linewidth=width)
plt.plot([40.00,60.00,80.00], [-4,-4,-4], '.k', linewidth=width)

I.fig.savefig('../windowingPlots2/2-Current-ext.svg', transparent=True,
                bbox_inches = "tight")
# I.fig.savefig('/mnt/WORK/work/MPhil(CCS)/ThesisDefense/Batteries/2-Current.svg', transparent=True,
#                 bbox_inches = "tight")
# %%
#sns.set(rc={'figure.figsize':(16,8)})
T = sns.relplot(x='Test_Time(m)', y='Temperature (C)_1', kind="line",
                data=train_X[0], size=20, color='m', legend=False)
T.fig.set_size_inches(12, 8)
#plt.xlim(-100, 40000)
# plt.ylim(8, 14)
# plt.title('Single Cycle - Temperature - 3 hours', size=16)
plt.ylabel('(C)   ', size=16, rotation=0)
plt.xlabel('Time (minutes)', size=16)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(which='major', alpha=1)

width : float = 1.5
# Box 1
x1,x2,y1,y2  = 0, 10, 19, 21#9, 12
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box 2
x1,x2,y1,y2  = 2.50, 12.50, 18.8, 20.8#8.8, 11.8
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box 3
x1,x2,y1,y2  = 6.00, 16.00, 18.7, 20.7#8.7, 11.7
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box Last
x1,x2,y1,y2  = 115.0, 125.0, 19.2, 21.2#9.2, 12.2
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Middle dots
# plt.plot([2000,3000,4000], [11.5,11.5,11.5], '.k', linewidth=width)
# plt.plot([2000,3000,4000], [ 9.5, 9.5, 9.5], '.k', linewidth=width)
plt.plot([40.00,60.00,80.00], [20.5,20.7,20.9], '.k', linewidth=width)
plt.plot([40.00,60.00,80.00], [19.0,19.2,19.4], '.k', linewidth=width)

T.fig.savefig('../windowingPlots2/3-Temperature-ext.svg', transparent=True,
                bbox_inches = "tight")
# T.fig.savefig('/mnt/WORK/work/MPhil(CCS)/ThesisDefense/Batteries/3-Temperature.svg', transparent=True,
#                 bbox_inches = "tight")
# %% SoC Plot
temp = pd.DataFrame(data={
                'SoC' : train_Y[0][:,0]*100,
                'Test_Time(m)' : train_X[0]['Test_Time(m)']
            }, dtype=np.float32)
S = sns.relplot(x='Test_Time(m)', y='SoC', kind="line",
                data=temp, size=20, color='k', legend=False)
S.fig.set_size_inches(12, 8)
# plt.plot(train_X[0]['Test_Time(m)'], train_Y[0]*100, linewidth=1.8, color='k')
# plt.title('Single Cycle - State of Charge - 3 hours', size=16)
plt.ylabel('(%)', size=16, rotation=0)
plt.xlabel('Time (minutes)', size=16)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(which='major', alpha=1)

width : float = 1.5
# Box 1
x1,x2,y1,y2  = 0, 10, 85, 105#9, 12
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box 2
x1,x2,y1,y2  = 2.50, 12.50, 80, 100#8.8, 11.8
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box 3
x1,x2,y1,y2  = 6.00, 16.00, 75, 95#8.7, 11.7
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box Last
x1,x2,y1,y2  = 115.0, 125.0, -5, 15#9.2, 12.2
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Middle dots
# plt.plot([2000,3000,4000], [11.5,11.5,11.5], '.k', linewidth=width)
# plt.plot([2000,3000,4000], [ 9.5, 9.5, 9.5], '.k', linewidth=width)
plt.plot([40.00,60.00,80.00], [80,65,50], '.k', linewidth=width)
plt.plot([40.00,60.00,80.00], [55,40,25], '.k', linewidth=width)

width : float = 3
plt.plot([10.00, 12.50, 16.00, 125.0], [94, 90, 88, 5.5], 'xk',
            linewidth=width, marker='v', markersize=11)
S.fig.savefig('../windowingPlots2/4-SoC-ext.svg', transparent=True,
                bbox_inches = "tight")
# plt.savefig('/mnt/WORK/work/MPhil(CCS)/ThesisDefense/Batteries/4-SoC.svg', transparent=True,
#                 bbox_inches = "tight")
# %%
# Windows i
plt.figure()
plt.plot(train_X[0]['Test_Time(m)'][0:500], train_X[0]['Voltage(V)'][0:500],
            color='r')
plt.title('V', size=60)
plt.xticks(size=17)
plt.yticks(size=17)
plt.ylabel('i        ', size=60, rotation=0)
plt.savefig('../windowingPlots2/i-V.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(m)'][0:500], train_X[0]['Current(A)'][0:500],
            color='b')
plt.title('I', size=60)
plt.xticks(size=17)
plt.yticks(size=17)
plt.savefig('../windowingPlots2/i-I.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(m)'][0:500], train_X[0]['Temperature (C)_1'][0:500],
            color='m')
plt.title('T', size=60)
plt.xticks(size=17)
plt.yticks(size=17)
plt.savefig('../windowingPlots2/i-T.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(temp['Test_Time(m)'][0:500], temp['SoC'][0:500],
            color='k')
plt.title('SoC', size=60)
plt.xticks(size=17)
plt.yticks(size=17)
plt.savefig('../windowingPlots2/i-S.svg', transparent=True,
                bbox_inches = "tight")
# %%
# Windows i+s
s : int = 250
plt.figure()
plt.plot(train_X[0]['Test_Time(m)'][0+s:500+s], train_X[0]['Voltage(V)'][0+s:500+s],
            color='r')
plt.xticks(size=17)
plt.yticks(size=17)
plt.ylabel('i+s      ', size=60, rotation=0)
plt.savefig('../windowingPlots2/i+s-V.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(m)'][0+s:500+s], train_X[0]['Current(A)'][0+s:500+s],
            color='b')
plt.xticks(size=17)
plt.yticks(size=17)
plt.savefig('../windowingPlots2/i+s-I.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(m)'][0+s:500+s], train_X[0]['Temperature (C)_1'][0+s:500+s],
            color='m')
plt.xticks(size=17)
plt.yticks(size=17)
plt.savefig('../windowingPlots2/i+s-T.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(temp['Test_Time(m)'][0+s:500+s], temp['SoC'][0+s:500+s],
            color='k')
plt.xticks(size=17)
plt.yticks(size=17)
plt.savefig('../windowingPlots2/i+s-S.svg', transparent=True,
                bbox_inches = "tight")
# %%
# Windows i+2s
s : int = 2*250
plt.figure()
plt.plot(train_X[0]['Test_Time(m)'][0+s:500+s], train_X[0]['Voltage(V)'][0+s:500+s],
            color='r')
plt.xticks(size=17)
plt.yticks(size=17)
plt.ylabel('i+2s      ', size=60, rotation=0)
plt.savefig('../windowingPlots2/i+2s-V.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(m)'][0+s:500+s], train_X[0]['Current(A)'][0+s:500+s],
            color='b')
plt.xticks(size=17)
plt.yticks(size=17)
plt.savefig('../windowingPlots2/i+2s-I.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(m)'][0+s:500+s], train_X[0]['Temperature (C)_1'][0+s:500+s],
            color='m')
plt.xticks(size=17)
plt.yticks(size=17)
plt.savefig('../windowingPlots2/i+2s-T.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(temp['Test_Time(m)'][0+s:500+s], temp['SoC'][0+s:500+s],
            color='k')
plt.xticks(size=17)
plt.yticks(size=17)
plt.savefig('../windowingPlots2/i+2s-S.svg', transparent=True,
                bbox_inches = "tight")
# %%
# Windows i+ns
plt.figure()
plt.plot(train_X[0]['Test_Time(m)'][-500:], train_X[0]['Voltage(V)'][-500:],
            color='r')
plt.xticks(size=17)
plt.yticks(size=17)
plt.ylabel('i+ns      ', size=60, rotation=0)
plt.savefig('../windowingPlots2/i+ns-V.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(m)'][-500:], train_X[0]['Current(A)'][-500:],
            color='b')
plt.xticks(size=17)
plt.yticks(size=17)
plt.savefig('../windowingPlots2/i+ns-I.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(m)'][-500:], train_X[0]['Temperature (C)_1'][-500:],
            color='m')
plt.xticks(size=17)
plt.yticks(size=17)
plt.savefig('../windowingPlots2/i+ns-T.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(temp['Test_Time(m)'][-500:], temp['SoC'][-500:],
            color='k')
plt.xticks(size=17)
plt.yticks(size=17)
plt.savefig('../windowingPlots2/i+ns-S.svg', transparent=True,
                bbox_inches = "tight")
# %%