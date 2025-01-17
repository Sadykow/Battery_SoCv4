from matplotlib.pyplot import subplots, close, FuncFormatter
from numpy import ndarray, linspace
from pandas import DataFrame

def format_SoC(value, _):
  return int(value*100)

def predicting_plot(profile : str, file_name : str, model_loc : str,
                    model_type : str,  iEpoch : str,
                    Y : ndarray, PRED : ndarray, RMS : ndarray,
                    val_perf : ndarray, TAIL : int,
                    save_plot : bool = False, RMS_plot : bool = True) -> None:
  """ Plot generator which is used for reporting results of the training for
  the article of Journal Article - Analysis.

  Args:
    profile (str): Profiler range saved in title.
    file_name (str): The name of the model. Either model number or Authors
  name.
    model_loc (str): Location for saving all modelps.
    model_type (str): LSTM or GRU name as a string.
    iEpoch (int): Number of iteration model went through.
    Y (ndarray): Y true array plot.
    PRED (ndarray): Y pred array for plot.
    RMS (ndarray): RMS result calculation.
    val_perf (ndarray): Results of the eval() function with 3 metrics.
    TAIL (int): The length of the prediction.
  """
  # Time range
  test_time = linspace(0, PRED.shape[0]/60, PRED.shape[0])
  
  # instantiate the first axes
  fig, ax1 = subplots(figsize=(14,12), dpi=600)
  ax1.plot(test_time[:TAIL:], Y[::,], '-',
          label="Actual", color='#0000ff')
  ax1.plot(test_time[:TAIL:],
          PRED, '--',
          label="Prediction", color='#ff0000')
  # ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
  ax1.set_xlabel("Time Slice (min)", fontsize=32)
  ax1.set_ylabel("SoC (%)", fontsize=32)
  
  # instantiate a second axes that shares the same x-axis
  if RMS_plot:
    ax2 = ax1.twinx()
    ax2.plot(test_time[:TAIL:],
          RMS,
          label="ABS error", color='#698856')
    ax2.fill_between(test_time[:TAIL:],
          RMS,
          color='#698856')
    ax2.set_ylabel('Error', fontsize=32, color='#698856')
    ax2.tick_params(axis='y', labelcolor='#698856', labelsize=28)
    ax2.set_ylim([-0.1,1.6])
    ax2.legend(loc='center right', bbox_to_anchor=(1.0,0.80), prop={'size': 32})
  ax1.set_title(
      f"{file_name} {model_type}. {profile}-trained",
      fontsize=36)
  ax1.legend(prop={'size': 32})
  ax1.tick_params(axis='both', labelsize=28)
  ax1.yaxis.set_major_formatter(FuncFormatter(format_SoC))
  ax1.set_ylim([-0.1,1.2])
  fig.tight_layout()

  # Put the text box with performance results.
  # textstr = '\n'.join((
  #     r'$MAE =%.2f$'  % (val_perf[1]*100, ),
  #     r'$RMSE=%.2f$'  % (val_perf[2]*100, ),
  #     r'$R^2 =%.2f$'  % (val_perf[3]*100, )))
  textstr = '\n'.join((
       '$MAE  = {0:.2f}%$'.format(val_perf[1]*100, ),
       '$RMSE = {0:.2f}%$'.format(val_perf[2]*100, ),
       '$R2  = {0:.2f}%$'.format(val_perf[3]*100, ) ))
  ax1.text(0.66, 0.74, textstr, transform=ax1.transAxes, fontsize=30,
          verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
  
  # Saving figure and cleaning Memory from plots
  if save_plot:
    fig.savefig(f'{model_loc}{profile}-{iEpoch}.svg')
  fig.clf()
  close()

def history_plot(profile : str, file_name : str, model_loc : str,
                 df : DataFrame, save_plot : bool = False,
                 metrics : list = ['mae', 'train_mae',
                                   'rmse', 'train_rms'],
                 plot_file_name : str = 'history.svg') -> None:
  fig, (ax1, ax2) = subplots(1,2, figsize=(28,12), dpi=600)
  fig.suptitle(f'{file_name} - {profile} training profile benchmark',
              fontsize=36)
  
  # Plot MAE subfigure
  ax1.plot(df[metrics[0]]*100, '-o',
      label="Training", color='#0000ff')
  ax1.plot(df[metrics[1]]*100, '--o',
      label="Validation", color='#0000ff')
  ax1.set_xlabel("Epochs", fontsize=32)
  ax1.set_ylabel("Error (%)", fontsize=32)
  ax2.set_xlabel("Epochs", fontsize=32)
  ax2.set_ylabel("Error (%)", fontsize=32)

  # Plot RMSE subfigure
  ax2.plot(df[metrics[2]]*100, '-o',
      label="Training", color='#ff0000')
  ax2.plot(df[metrics[3]]*100, '--o',
      label="Validation", color='#ff0000')
  ax1.set_ylabel("Error (%)", fontsize=32)
  ax1.legend(prop={'size': 32})
  ax2.legend(prop={'size': 32})
  ax1.tick_params(axis='both', labelsize=28)
  ax2.tick_params(axis='both', labelsize=28)

  # Tighting the layot
  ax1.set_title(f"Mean Absoulute Error", fontsize=36)
  ax2.set_title(f"Root Mean Squared Error", fontsize=36)
  ax1.set_ylim([-0.1,11])
  ax2.set_ylim([-0.1,11])
  fig.tight_layout()
  
  # Saving figure and cleaning Memory from plots
  if save_plot:
    fig.savefig(f'{model_loc}{plot_file_name}')
  fig.clf()
  close()