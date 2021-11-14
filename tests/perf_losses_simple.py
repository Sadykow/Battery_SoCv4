# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mpl.rcParams['figure.figsize'] = (14, 12)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['font.family'] = 'Bender'
# %%
profiles : list = ['DST', 'US06', 'FUDS']
authors  : list = [None, 'Chemali2017', 'BinXiao2020', 'TadeleMamo2020',
                         'MengJiao2020', 'GelarehJavid2020', 'WeiZhang2020']

# for author in authors:
#     for profile in profiles:

# %%
N = 5
for profile in profiles[2:]:
    df = pd.read_csv(f'../Models/{authors[N]}/{profile}-models/history-{profile}.csv')
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(28,12), dpi=600)
    fig.suptitle(f'Model №{N} - {profile} training profile benchmark', fontsize=36)
    ax1.plot(df['mae']*100, '-o',
            label="Training", color='#0000ff')
    ax1.plot(df['val_mean_absolute_error']*100, '--o',
            label="Validation", color='#0000ff')
    ax1.set_xlabel("Epochs", fontsize=32)
    ax1.set_ylabel("Error (%)", fontsize=32)
    ax2.set_xlabel("Epochs", fontsize=32)
    ax2.set_ylabel("Error (%)", fontsize=32)

    ax2.plot(df['rmse']*100, '-o',
            label="Training", color='#ff0000')
    ax2.plot(df['val_root_mean_squared_error']*100, '--o',
            label="Validation", color='#ff0000')
    ax1.set_ylabel("Error (%)", fontsize=32)
    ax1.legend(prop={'size': 32})
    ax2.legend(prop={'size': 32})
    ax1.tick_params(axis='both', labelsize=28)
    ax2.tick_params(axis='both', labelsize=28)

    ax1.set_title(
        f"Mean Absoulute Error",
        fontsize=36)
    ax2.set_title(
        f"Root Mean Squared Error",
        fontsize=36)
    ax1.set_ylim([-0.1,8])
    ax2.set_ylim([-0.1,11])
    # fig.tight_layout()
#     fig.savefig(f'../Models/{authors[N]}/{profile}-models/{authors[N][:2]}history-{profile}.svg')
# %%