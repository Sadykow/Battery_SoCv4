# %%
import os, time                   # Moving to folders in FS and timing
import pandas as pd               # Tabled data storage
import numpy as np

# to measure exec time
from itertools import chain       # Make Chain ranges
from numba import jit, float32, cuda

columns = ['Current(A)', 'Voltage(V)', 'Temperature (C)_1',
		   'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
		]
# %%
@jit(float32(float32, float32))
def diffSoC(chargeData, discargeData):
	return np.round((chargeData - discargeData)*100)/100

def Read_Excel_File(path : str, indexes : range) -> pd.DataFrame:
	if(indexes == None):
		indexes = chain(range(4 , 12), range(18, 25))
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
	df = df[df['Step_Index'].isin(indexes)]
	df = df.reset_index(drop=True)
	df = df.drop(columns=['Step_Index'])
	df = df[columns]   # Order columns in the proper sequence
	return df

@jit(parallel=True)
def ParseExcelData(directory : str,
                    indexes : range,
              ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
	""" Parsing Excel data from Battery Testing Machine

	Args:
	directory (str): Dataset directory location. !! Make sure not other file
	formats stored. No check has been added.
	indexes (range): The data indexes to select regions

	Returns:
	tuple[list[pd.DataFrame], list[pd.DataFrame]]: Returning data itself 
	and SoC in the list format to capture absolutely all samples.
	"""
	for _, _, files in os.walk(directory):
		files.sort(key=lambda f: int(f[-13:-5])) # Sort by last dates
		# Initialize empty structures
		data_df : list[pd.DataFrame] = []
		data_SoC : list[pd.DataFrame] = []
		for file in files[:]:
			df : pd.DataFrame = Read_Excel_File(directory + '/' + file,
					indexes)
			SoC: pd.DataFrame = pd.DataFrame(
				data={'SoC' : diffSoC(
								chargeData=np.array(df.loc[:,'Charge_Capacity(Ah)'], dtype=np.float32),
								discargeData=np.array(df.loc[:,'Discharge_Capacity(Ah)'], dtype=np.float32)
								)},
				dtype=np.float32
				)
			data_df.append(df)
			data_SoC.append(SoC)
		return data_df, data_SoC
# %%
if __name__=="__main__":
	train_dir	  : str ='../Data/A123_Matt_Set'
	r_DST_US_FUDS : range = range(4 , 25)  # Full cycle
	tic : float = time.perf_counter()
	tr_ls_df, tr_ls_SoC = ParseExcelData(train_dir, None)
	print(f'Time: {time.perf_counter() - tic}')
    # n = 10000000							
	# a = np.ones(n, dtype = np.float64)
	# b = np.ones(n, dtype = np.float32)
	
	# start = timer()
	# func(a)
	# print("without GPU:", timer()-start)	
	
	# start = timer()
	# func2(a)
	# print("with GPU:", timer()-start)
# %%