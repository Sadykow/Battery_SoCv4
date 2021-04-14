# %%
import os, sys
import pandas as pd  # File read
import matplotlib.pyplot as plt

from time import perf_counter


sys.path.append(os.getcwd() + '/..')
from extractor.DataGenerator import *
from extractor.WindowGenerator import WindowGenerator

# %%
if __name__ == '__main__':
    Data    : str = '../Data/'
    dataGenerator = DataGenerator(train_dir=f'{Data}A123_Matt_Set',
                                valid_dir=f'{Data}A123_Matt_Val',
                                test_dir=f'{Data}A123_Matt_Test',
                                columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                                'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                                    ],
                                PROFILE_range = 'FUDS')
# %%
    window1 = WindowGenerator(Data=dataGenerator,
                        input_width=500, label_width=1, shift=0,
                        input_columns=['Current(A)', 'Voltage(V)',
                                                'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,
                        includeTarget=False, normaliseLabal=False,
                        shuffleTraining=False)
    window2 = WindowGenerator(Data=dataGenerator,
                        input_width=500, label_width=1, shift=1,
                        input_columns=['Current(A)', 'Voltage(V)',
                                                'Temperature (C)_1'],
                        label_columns=['SoC(%)'], batch=1,
                        includeTarget=False, normaliseLabal=False,
                        shuffleTraining=False)
# %%
    #tic = perf_counter()
    print('Windowing 1')
    ds, x, y = window1.make_dataset_from_array(
                                  inputs=dataGenerator.valid,
                                  labels=dataGenerator.valid_SoC
                                )
    #print(f'Windowing Array took {perf_counter()-tic}')
# %%
    print('Windowing 1')
    xx, yy = window1.make_dataset_from_list(
                                  X=dataGenerator.valid_list,
                                  Y=dataGenerator.valid_list_label,
                                  look_back  = 500
                                )

# %%
    test_y = y[:yy[0].shape[0]]
    test_yy = yy[0]
    plt.plot(test_y)
    plt.plot(test_yy)
    count = 0
    for i in range(len(test_yy)):
        if(test_yy[i] != test_y[i]):
            count +=1
    print(count)
# %%
    test_x = x[:yy[0].shape[0]]
    test_xx = xx[0]
    plt.plot(test_x[:,0,0])
    plt.plot(test_xx[:,0,0])
    
    count = 0
    for i in range(len(test_x[0,:,0])):
        if(test_x[0,i,0] != test_xx[0,i,0]):
            count +=1
    print(count)
# %%