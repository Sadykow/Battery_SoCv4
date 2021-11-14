# Accyracy over time plot as per root squared error ny Chemali2017
# %%
import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # File read

# %%
x = [200, 250, 375, 500, 1000, 1500]
#1.303-(1.303-0.672)/(1+1/4)
#0.573-(0.672-0.573)/(2+1/2)
y = [1.5, 1.303, 0.7982, 0.672, 0.573, 0.5334]


#1.670-(1.670-0.863)/(1+1/4)
#0.690-(0.863-0.690)/(2+1/2)
rms_y = [2.0, 1.670, 1.0244, 0.863, 0.690, 0.6207]
fit = np.polyfit(x, y, 2)
a = fit[0]
b = fit[1]
c = fit[2]

# fit_equation = a * np.square(x) + b * x + c
fit_equation = a * np.sqrt(x+b) + c
# fit_equation = a * np.sqrt(x) + c
#Plotting
fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
# ax1.plot(x, fit_equation,color = 'r',alpha = 0.5, label = 'Polynomial fit')
# ax1.scatter(x, y,s = 5, color = 'r', label = 'Data points')
# MAE
ax1.plot(x, y, '-', color='#0000ff', label = 'MAE(%)')
ax1.plot([250, 500, 1000], [1.303, 0.672, 0.573], 'bx')
ax1.text(140,1.263,'1.303', fontsize=24)
ax1.text(500,0.7,'0.672', fontsize=24)
ax1.text(1000,0.6,'0.573', fontsize=24)
# RMS
ax1.plot(x, rms_y, '--', color='#ff0000', label = 'RMS(%)')
ax1.plot([250, 500, 1000], [1.670, 0.863, 0.690], 'rx')
ax1.text(260,1.670,'1.670', fontsize=24)
ax1.text(500,0.893,'0.863', fontsize=24)
ax1.text(1000,0.720,'0.690', fontsize=24)

ax1.set_title('Accuracy of LSTM-RNN with varios network depths', fontsize=36)
ax1.set_xlabel('Network depth in time', fontsize=32)
ax1.set_ylabel('Error', fontsize=32)
ax1.set_ylim([0.50, 1.77])
ax1.legend(prop={'size': 32})
ax1.tick_params(axis='both', labelsize=28)
fig.tight_layout()
plt.show()
fig.savefig(f'figures/accuracy.svg')
# %%
# %%
x = np.linspace(0,1200,1200)
y = -1/4*np.sqrt(x)+10
plt.plot(x,y)
plt.grid()
plt.xlabel('Network depth in time')
plt.ylabel('MAE(%)')
plt.title('Accuracy of LSTM-RNN with varios network depths on Time')
plt.ylim([0,15])
plt.yticks([], [])
plt.plot([250,500,1000], [6,4.5,2], 'rx')
plt.text(250,7,'1.303')
plt.text(500,5,'0.672')
plt.text(1000,2.5,'0.573')
plt.savefig('accuracy.png')
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x,a,c):
    print('%.3f, %.3f' % (a,c))
    return a * np.sqrt(x) + c

# x = np.array([5, 11, 15, 44, 60, 70, 75, 100, 120, 200])
# y_true = np.array([2.492, 8.330, 11.000, 19.394, 24.466, 27.777, 29.878, 26.952, 35.607, 46.966])
x = np.array([1, 2, 3, 4])
y_true = np.array([1.303, 0.672, 0.573, 0.52349])

popt, pcov = curve_fit(func,x,y_true)
popt = [2.252, 5.000, 6.908]
y_pred = func(x,*popt)

fig,ax = plt.subplots(figsize=(8,6))
ax.scatter(x,y_true,c='r',label='true',s=6)
ax.plot(x,y_pred,c='g',label='pred')
ax.legend(loc='best')