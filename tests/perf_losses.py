# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# %%
y_pre_values = [0.86, 0.64, 0.57, 0.53, 0.48, 0.42, 0.38]
y_values = [0.38, 0.34, 0.25, 0.16, 0.08]

x_pre_ticks = range(0, len(y_pre_values))
x_ticks = range(6, 6+len(y_values))
# %%
plt.figure(figsize=(12,8))
plt.plot(x_pre_ticks, y_pre_values, '-')
plt.plot(x_ticks, y_values, '-x')
n_pre_ticks = 12
pre_ticks = np.round(np.linspace(0, 1, n_pre_ticks), 2)

def format_func(value, tick_number):
    if(tick_number < n_pre_ticks):
        return pre_ticks[tick_number]
    else:
        return float(value-5)

ax = plt.axes()
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xticks(range(11))
plt.grid()
plt.xticks(
        np.concatenate(
            (np.linspace(0,6, n_pre_ticks), np.arange(7,11,step=1)),
            axis=0)
    )
# %%