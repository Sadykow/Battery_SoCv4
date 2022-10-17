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