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

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.backend import shape

# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
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
# %% [markdown]
# # Performing Plots generation from 10C example
# %%
#sns.set(rc={'figure.figsize':(16,8)})
V = sns.relplot(x='Test_Time(s)', y='Voltage(V)', kind="line",
                data=train_X[0], size=20, color='r')
#plt.xlim(-100, 40000)
plt.ylim(2.25, 3.78)
plt.ylabel('Voltage (V)', size=16)
plt.xlabel('Time (s)', size=16)
plt.xticks(size=14)
plt.yticks(size=20)
plt.grid(which='major', alpha=1)
#plt.legend(False)
width : float = 1.5
# Box 1
x1,x2,y1,y2  = 0, 500, 2.6, 3.75
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box 2
x1,x2,y1,y2  = 250, 750, 2.55, 3.70
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box 3
x1,x2,y1,y2  = 600, 1100, 2.53, 3.68
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box Last
x1,x2,y1,y2  = 6500, 7000, 2.55, 3.4
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Middle dots
plt.plot([2000,3000,4000], [3.62,3.6,3.58], '.k', linewidth=width)
plt.plot([2000,3000,4000], [2.72,2.7,2.68], '.k', linewidth=width)
#g.fig.autofmt_xdate()
# sns.boxplot(x='Test_Time(s)', y='Voltage(V)', data=train_X[0], 
#                  showcaps=False,boxprops={'facecolor':'None', "zorder":10},
#                  showfliers=False,whiskerprops={'linewidth':0, "zorder":10},
#                  ax=V, zorder=10)

V.fig.savefig('../windowingPlots/1-Voltage.svg', transparent=True,
                bbox_inches = "tight")
# %%
I = sns.relplot(x='Test_Time(s)', y='Current(A)', kind="line",
                data=train_X[0], size=20, color='b')
#plt.xlim(-100, 40000)
#plt.ylim(2.25, 3.75)
#g.fig.autofmt_xdate()
plt.ylabel('Current (A)', size=16)
plt.xlabel('Time (s)', size=16)
plt.xticks(size=12)
plt.yticks(size=20)
plt.grid(which='major', alpha=1)
#plt.legend(False)
width : float = 1.5
# Box 1
x1,x2,y1,y2  = 0, 500, -4, 2
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box 2
x1,x2,y1,y2  = 250, 750, -3.9, 2.1
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box 3
x1,x2,y1,y2  = 600, 1100, -4.1, 1.9
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box Last
x1,x2,y1,y2  = 6500, 7000, -4, 2
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Middle dots
plt.plot([2000,3000,4000], [ 2, 2, 2], '.k', linewidth=width)
plt.plot([2000,3000,4000], [-4,-4,-4], '.k', linewidth=width)

I.fig.savefig('../windowingPlots/2-Current.svg', transparent=True,
                bbox_inches = "tight")

# %%
#sns.set(rc={'figure.figsize':(16,8)})
T = sns.relplot(x='Test_Time(s)', y='Temperature (C)_1', kind="line",
                data=train_X[0], size=20, color='m')
#plt.xlim(-100, 40000)
# plt.ylim(8, 14)
plt.ylabel('Temperature (C)', size=16)
plt.xlabel('Time (s)', size=16)
plt.xticks(size=12)
plt.yticks(size=20)
plt.grid(which='major', alpha=1)
#plt.legend(False)
width : float = 1.5
# Box 1
x1,x2,y1,y2  = 0, 500, 19, 21#9, 12
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box 2
x1,x2,y1,y2  = 250, 750, 18.8, 20.8#8.8, 11.8
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box 3
x1,x2,y1,y2  = 600, 1100, 18.7, 20.7#8.7, 11.7
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Box Last
x1,x2,y1,y2  = 6500, 7000, 19.2, 21.2#9.2, 12.2
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Middle dots
# plt.plot([2000,3000,4000], [11.5,11.5,11.5], '.k', linewidth=width)
# plt.plot([2000,3000,4000], [ 9.5, 9.5, 9.5], '.k', linewidth=width)
plt.plot([2000,3000,4000], [20.5,20.7,20.9], '.k', linewidth=width)
plt.plot([2000,3000,4000], [19.0,19.2,19.4], '.k', linewidth=width)

T.fig.savefig('../windowingPlots/3-Temperature.svg', transparent=True,
                bbox_inches = "tight")
# %% SoC Plot
T = sns.relplot(x='Test_Time(s)', y='SoC', kind="line",
                data=Y, size=20, color='k')
plt.ylabel('SoC', size=16)
plt.xlabel('Time (s)', size=16)
plt.xticks(size=12)
plt.yticks(size=20)
plt.grid(which='major', alpha=1)
#plt.legend(False)
width : float = 3
plt.plot([500, 750,1100,7000], [0.94,0.90,0.86,0.075], 'xk',
            linewidth=width, marker='v', markersize=11)
T.fig.savefig('../windowingPlots/4-SoC.svg', transparent=True,
                bbox_inches = "tight")
# %%
# Windows i
plt.figure()
plt.plot(train_X[0]['Test_Time(s)'][0:500], train_X[0]['Voltage(V)'][0:500],
            color='r')
plt.title('V', size=60)
plt.xticks(size=12)
plt.yticks(size=12)
plt.ylabel('i        ', size=60, rotation=0)
plt.savefig('../windowingPlots/i-V.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(s)'][0:500], train_X[0]['Current(A)'][0:500],
            color='b')
plt.title('I', size=60)
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('../windowingPlots/i-I.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(s)'][0:500], train_X[0]['Temperature (C)_1'][0:500],
            color='k')
plt.title('T', size=60)
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('../windowingPlots/i-T.svg', transparent=True,
                bbox_inches = "tight")
# %%
# Windows i+s
s : int = 250
plt.figure()
plt.plot(train_X[0]['Test_Time(s)'][0+s:500+s], train_X[0]['Voltage(V)'][0+s:500+s],
            color='r')
plt.xticks(size=12)
plt.yticks(size=12)
plt.ylabel('i+s      ', size=60, rotation=0)
plt.savefig('../windowingPlots/i+s-V.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(s)'][0+s:500+s], train_X[0]['Current(A)'][0+s:500+s],
            color='b')
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('../windowingPlots/i+s-I.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(s)'][0+s:500+s], train_X[0]['Temperature (C)_1'][0+s:500+s],
            color='k')
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('../windowingPlots/i+s-T.svg', transparent=True,
                bbox_inches = "tight")
# %%
# Windows i+2s
s : int = 2*250
plt.figure()
plt.plot(train_X[0]['Test_Time(s)'][0+s:500+s], train_X[0]['Voltage(V)'][0+s:500+s],
            color='r')
plt.xticks(size=12)
plt.yticks(size=12)
plt.ylabel('i+2s      ', size=60, rotation=0)
plt.savefig('../windowingPlots/i+2s-V.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(s)'][0+s:500+s], train_X[0]['Current(A)'][0+s:500+s],
            color='b')
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('../windowingPlots/i+2s-I.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(s)'][0+s:500+s], train_X[0]['Temperature (C)_1'][0+s:500+s],
            color='k')
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('../windowingPlots/i+2s-T.svg', transparent=True,
                bbox_inches = "tight")
# %%
# Windows i+ns
plt.figure()
plt.plot(train_X[0]['Test_Time(s)'][6500:7000], train_X[0]['Voltage(V)'][6500:7000],
            color='r')
plt.xticks(size=12)
plt.yticks(size=12)
plt.ylabel('i+ns      ', size=60, rotation=0)
plt.savefig('../windowingPlots/i+ns-V.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(s)'][6500:7000], train_X[0]['Current(A)'][6500:7000],
            color='b')
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('../windowingPlots/i+ns-I.svg', transparent=True,
                bbox_inches = "tight")
plt.figure()
plt.plot(train_X[0]['Test_Time(s)'][6500:7000], train_X[0]['Temperature (C)_1'][6500:7000],
            color='k')
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('../windowingPlots/i+ns-T.svg', transparent=True,
                bbox_inches = "tight")
# %%