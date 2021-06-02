# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %%
mpl.rcParams['figure.figsize'] = (14, 12)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['font.family'] = 'Bender'
df = pd.read_csv('/mnt/WORK/QUT/TF/Battery_SoCv4/Models/Chemali2017/DST-models/history-DST.csv')
y_pre_values = [0.086, 0.064, 0.057, 0.053, 0.048, 0.046, 0.045]
# y_values = [0.38, 0.34, 0.25, 0.16, 0.08]
y_values = df['mean_absolute_error'][:30]

x_pre_ticks = range(0, len(y_pre_values))
x_ticks = range(len(x_pre_ticks)-1, len(x_pre_ticks)-1+len(y_values))
# %%
n_pre_ticks = 12
pre_ticks = np.round(np.linspace(0, 1, n_pre_ticks), 2)

def format_func(value, tick_number):
    if(tick_number < n_pre_ticks):
        return pre_ticks[tick_number]
    else:
        return int(value-5)

fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
ax1.plot(x_pre_ticks, y_pre_values, '-')
ax1.plot(x_ticks, y_values, '-x')

ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
ax1.set_xlabel("Sample", fontsize=32)
ax1.set_ylabel("MAE(%)", fontsize=32)
ax1.set_title(
    f"Title-trained",
    fontsize=36)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# plt.xticks(range(11))
plt.xticks(
        np.concatenate(
            (np.linspace(0,6, n_pre_ticks), np.arange(7,len(y_values)+6,step=1)),
            axis=0)
    )
# ax1.tick_params(np.concatenate(
#                 np.linspace(0,6, n_pre_ticks), np.arange(7,11,step=1)
#             ), axis='x'
#         )
fig.savefig('figures/loss1.svg')
# %%
fig, ax1 = plt.subplots(figsize=(14,12), dpi=600)
ax1.plot(range(0,30), y_values, '-')

ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
ax1.set_xlabel("Sample", fontsize=32)
ax1.set_ylabel("MAE(%)", fontsize=32)
ax1.set_xticks(np.arange(0, len(x_ticks), 1))
ax1.set_title(
    "Epochs",
    fontsize=36)

ax2 = ax1.twiny()
ax2.plot(range(0,30,5), y_values[::5], 'x', label="RMS error", color='#698856')
# ax2.plot(test_time[:TAIL:],
#         RMS,
#         label="RMS error", color='#698856')
# ax2.fill_between(test_time[:TAIL:],
#         RMS[:,0],
#         color='#698856')
ax2.set_ylabel('Error', fontsize=32, color='#698856')
ax2.tick_params(axis='y', labelcolor='#698856', labelsize=24)
# ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# # plt.xticks(range(11))
# plt.xticks(
#         np.concatenate(
#             (np.linspace(0,6, n_pre_ticks), np.arange(7,len(y_values),step=1)),
#             axis=0)
#     )
fig.savefig('figures/loss3.svg')
# %%