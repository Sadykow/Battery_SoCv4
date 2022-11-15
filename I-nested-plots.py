#!/usr/bin/python
# %%
from turtle import title
import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # File read

from extractor.DataGenerator import *

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# %%
Data = 'Data/'
profile = 'FUDS'
formats = 'svg'
dataGenerator = DataGenerator(train_dir=f'{Data}A123_Matt_Set',
                              valid_dir=f'{Data}A123_Matt_Val',
                              test_dir=f'{Data}A123_Matt_Test',
                              columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                                'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                                ],
                              PROFILE_range = profile,
                              round=5)
# %%
#? DST
current = dataGenerator.valid[:,0]
t_currt = np.arange(len(current))
lng = 365
x1 = 5800
x2 = x1 + lng # 6165
width : float = 1.5
split = range(x1, x2)
# Original plot
fig = plt.figure()
plt.plot(t_currt, current)# First new axes
plt.title(f'Interpolated current of a single cycle {profile} profile '
            'sampled at 1Hz'
            )
plt.ylabel('Current(A)')
plt.xlabel('time(s)')
plt.ylim([-4.5, 2.5])
# Box 1
y1,y2  = -4.1, 2.1
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Background
# x_points = [1.5, 2.6, 3.5, 4, 9]
y_points = [2, 1, 0, -1, -2, -3, -4]
for i in range(len(y_points)):
    plt.plot([x1,x2], [y_points[i], y_points[i]], linestyle='dashed', color='gray', linewidth=0.8)

#              [left, bottom, width, height]
ax = fig.add_axes([0.18, 0.2, 0.25, 0.35])
#plt.plot(t_currt[split], aCurren[split])
ax.set_title(f'Repeated subcycle - {lng}s')
ax.plot(t_currt[split], current[split])
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.set_ylim([-4.2, 2.2])
ax.set_xticks(np.arange(x1,x2,80))

# fig.savefig(f'Modds/Current-{profile}.{formats}', transparent=True)
# plt.show()

# %%
#? US06
current = dataGenerator.valid[:,0]
t_currt = np.arange(len(current))
lng = 600
x1 = 5290
x2 =  x1+lng
width : float = 1.5
split = range(x1, x2)
# Original plot
fig = plt.figure()
plt.plot(t_currt, current)# First new axes
plt.title(f'Interpolated input current of a single cycle {profile} profile '
            'sampled at 1Hz'
            )
plt.ylabel('Current(A)')
plt.xlabel('time(s)')
plt.ylim([-4.5, 2.5])
# Box 1
y1,y2  = -4.1, 1.1
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Background
# x_points = [1.5, 2.6, 3.5, 4, 9]
y_points = [1, 0, -1, -2, -3, -4]
for i in range(len(y_points)):
    plt.plot([x1,x2], [y_points[i], y_points[i]], linestyle='dashed', color='gray', linewidth=0.8)

#              [left, bottom, width, height]
ax = fig.add_axes([0.18, 0.2, 0.25, 0.35])
#plt.plot(t_currt[split], aCurren[split])
ax.set_title(f'Repeated subcycle - {lng}s')
ax.plot(t_currt[split], current[split])
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.set_ylim([-4.2, 2.2])
ax.set_xticks(np.arange(x1,x2,130))

# fig.savefig(f'Modds/Current-{profile}.{formats}', transparent=True)
# plt.show()
# %%
#? FUDS
current = dataGenerator.valid[:,0]
t_currt = np.arange(len(current))
lng = 1400
x1 = 6080
x2 =  x1+lng
width : float = 1.5
split = range(x1, x2)
# Original plot
fig = plt.figure()
plt.plot(t_currt, current)# First new axes
plt.title(f'Interpolated input current of a single cycle {profile} profile '
            'sampled at 1Hz'
            )
plt.ylabel('Current(A)')
plt.xlabel('time(s)')
plt.ylim([-4.5, 2.5])
# Box 1
y1,y2  = -4.1, 2.1
plt.plot([x1,x1], [y1,y2], linewidth=width, color='k')
plt.plot([x2,x2], [y1,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y2,y2], linewidth=width, color='k')
plt.plot([x1,x2], [y1,y1], linewidth=width, color='k')

# Background
# x_points = [1.5, 2.6, 3.5, 4, 9]
y_points = [2, 1, 0, -1, -2, -3, -4]
for i in range(len(y_points)):
    plt.plot([x1,x2], [y_points[i], y_points[i]], linestyle='dashed', color='gray', linewidth=0.8)

#              [left, bottom, width, height]
ax = fig.add_axes([0.18, 0.2, 0.25, 0.35])
#plt.plot(t_currt[split], aCurren[split])
ax.set_title(f'Repeated subcycle - {lng}s')
ax.plot(t_currt[split], current[split])
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.set_ylim([-4.2, 2.2])
ax.set_xticks(np.arange(x1,x2,310))

# fig.savefig(f'Modds/Current-{profile}.{formats}', transparent=True)
# plt.show()
# %%
#[DEmo plot of data split]
dataDST = DataGenerator(train_dir=f'{Data}A123_Matt_Set',
                              valid_dir=f'{Data}A123_Matt_Val',
                              test_dir=f'{Data}A123_Matt_Test',
                              columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                                'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                                ],
                              PROFILE_range = 'DST',
                              round=5)
dataUS = DataGenerator(train_dir=f'{Data}A123_Matt_Set',
                              valid_dir=f'{Data}A123_Matt_Val',
                              test_dir=f'{Data}A123_Matt_Test',
                              columns=[
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                                'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                                ],
                              PROFILE_range = 'US06',
                              round=5)

soc_fuds = dataGenerator.train_label[:,0]
t_fuds = np.arange(len(soc_fuds))
soc_dst = dataDST.train_label[:,0]
t_dst = np.arange(len(soc_dst))
soc_us= dataUS.train_label[:,0]
t_us = np.arange(len(soc_us))
# %%
fig = plt.figure()
axs = fig.subplots(3,1)
axs[0].set_title('Model fitting data split')

axs[0].plot(t_fuds, soc_fuds)
axs[0].text(4000,  0.1, '20$^\circ$C', fontsize=10)
axs[0].text(15000, 0.1, '25$^\circ$C', fontsize=10)
axs[0].text(27000, 0.1, '30$^\circ$C', fontsize=10)
axs[0].text(39000, 0.1, '40$^\circ$C', fontsize=10)
axs[0].text(51000, 0.1, '50$^\circ$C', fontsize=10)
axs[0].set_ylim([-0.05, 1.22])
axs[0].set_ylabel('Cycle Type \nTrain')
axs[0].tick_params(left = False, bottom = False, 
                   labelleft = False, labelbottom = False,)
#train
axs[0].text(24000, 1.03, 'Train', fontsize=16)
axs[0].annotate('', xy=(0,0.9), xytext=(t_fuds[-1],0.9),
                 arrowprops=dict(arrowstyle='<->'))
axs[0].plot([0, 0], [0, 1], linestyle="dashed", color='k')
axs[0].plot([t_fuds[-1], t_fuds[-1]], [0, 1], linestyle="dashed", color='k')
#valid
axs[0].text(14000, 0.36, 'Valid', fontsize=16)
axs[0].annotate('', xy=(11000,0.33), xytext=(24000,0.33),
                 arrowprops=dict(arrowstyle='<->'))
axs[0].plot([11000, 11000], [0, 0.41], linestyle="dashed", color='k')
axs[0].plot([24000, 24000], [0, 0.41], linestyle="dashed", color='k')

axs[1].plot(t_dst, soc_dst)
axs[1].text(4000,  0.1, '20$^\circ$C', fontsize=10)
axs[1].text(16000, 0.1, '25$^\circ$C', fontsize=10)
axs[1].text(28000, 0.1, '30$^\circ$C', fontsize=10)
axs[1].text(40000, 0.1, '40$^\circ$C', fontsize=10)
axs[1].text(52000, 0.1, '50$^\circ$C', fontsize=10)
axs[1].set_ylim([-0.05, 1.22])
axs[1].set_ylabel('Cycle Type \nTest 1')
axs[1].tick_params(left = False, bottom = False, 
                   labelleft = False, labelbottom = False,)
axs[1].text(21000, 1.03, 'Test', fontsize=16)
axs[1].annotate('', xy=(12000,0.9), xytext=(36000,0.9),
                 arrowprops=dict(arrowstyle='<->'))
axs[1].plot([12000, 12000], [0, 1], linestyle="dashed", color='k')
axs[1].plot([36000, 36000], [0, 1], linestyle="dashed", color='k')


axs[2].plot(t_us, soc_us)
axs[2].text(4000,  0.1, '20$^\circ$C', fontsize=10)
axs[2].text(15000, 0.1, '25$^\circ$C', fontsize=10)
axs[2].text(26000, 0.1, '30$^\circ$C', fontsize=10)
axs[2].text(37000, 0.1, '40$^\circ$C', fontsize=10)
axs[2].text(50000, 0.1, '50$^\circ$C', fontsize=10)
axs[2].set_ylim([-0.05, 1.22])
axs[2].set_ylabel('Cycle Type \nTest 2')
axs[2].tick_params(left = False, bottom = False, 
                   labelleft = False, labelbottom = False,)
axs[2].text(20000, 1.03, 'Test', fontsize=16)
axs[2].annotate('', xy=(11500,0.9), xytext=(34000,0.9),
                 arrowprops=dict(arrowstyle='<->'))
axs[2].plot([11500, 11500], [0, 1], linestyle="dashed", color='k')
axs[2].plot([34000, 34000], [0, 1], linestyle="dashed", color='k')
fig.savefig(f'Modds/tmp/cross-data.svg', transparent=True)
# plt.title(f'Interpolated input current of a single cycle {profile} profile '
#             'sampled at 1Hz'
#             )
# plt.ylabel('Current(A)')
# plt.xlabel('time(s)')
# plt.ylim([-4.5, 2.5])