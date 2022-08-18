#!/usr/bin/python
# %% [markdown]
# # # 4
# # #
# # LSTM for SoC by GatethJavid 2020
# PyTorch version for optimiser and custom training loop with pre-existing
# optimiser
# %%
import datetime
import logging
import os, sys, getopt    # OS, SYS, argc functions
from sys import platform  # Get type of OS

import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
plt.switch_backend('agg')       #! FIX in the no-X env: RuntimeError: Invalid DISPLAY variable
import numpy as np
import pandas as pd  # File read
import torch
import torch.nn as nn
from torch.autograd import Variable
#from torch.utils.data import DataLoader
import torchmetrics

from tqdm import tqdm, trange

from extractor.DataGenerator import *
from py_modules.PT_RoAdam import RoAdam
from py_modules.utils import str2bool, Locate_Best_Epoch
from py_modules.plotting import predicting_plot, history_plot

from typing import Callable
if (sys.version_info[1] < 9):
    LIST = list
    from typing import List as list
    from typing import Tuple as tuple

import gc           # Garbage Collector
# %%
# Extract params
# try:
#     opts, args = getopt.getopt(sys.argv[1:],"hd:e:l:n:a:g:p:",
#                     ["help", "debug=", "epochs=", "layers=", "neurons=",
#                      "attempt=", "gpu=", "profile="])
# except getopt.error as err: 
#     # output error, and return with an error code 
#     print (str(err)) 
#     print ('EXEPTION: Arguments requied!')
#     sys.exit(2)

opts = [('-d', 'False'), ('-e', '100'), ('-l', '3'), ('-n', '131'), ('-a', '13'),
        ('-g', '0'), ('-p', 'FUDS')] # 2x131 1x1572 
debug   : int = 0
batch   : int = 1
mEpoch  : int = 10
nLayers : int = 1
nNeurons: int = 262
attempt : str = '1'
GPU     : int = None
profile : str = 'DST'
rounding: int = 5
print(opts)
for opt, arg in opts:
    if opt == '-h':
        print('HELP: Use following default example.')
        print('python *.py --debug False --epochs 50 --gpu 0 --profile DST')
        print('TODO: Create a proper help')
        sys.exit()
    elif opt in ("-d", "--debug"):
        if(str2bool(arg)):
            logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s --> %(levelname)s:%(message)s')
            logging.warning("Logger DEBUG")
            debug = 1
        else:
            logging.basicConfig(level=logging.CRITICAL)
            logging.warning("Logger Critical")
            debug = 0
    elif opt in ("-e", "--epochs"):
        mEpoch = int(arg)
    elif opt in ("-l", "--layers"):
        nLayers = int(arg)
    elif opt in ("-n", "--neurons"):
        nNeurons = int(arg)
    elif opt in ("-a", "--attempts"):
        attempt = (arg)
    elif opt in ("-g", "--gpu"):
        #! Another alternative is to use
        #!:$ export CUDA_VISIBLE_DEVICES=0,1 && python *.py
        GPU = int(arg)
    elif opt in ("-p", "--profile"):
        profile = (arg)
# %%
# Define plot sizes
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# Configurage logger and print basics
logging.basicConfig(level=logging.CRITICAL,        
    format='%(asctime)s --> %(levelname)s:%(message)s')
logging.warning("Logger enabled")

logging.debug("\n\n"
    f"MatPlotLib version: {mpl.__version__}\n"
    f"Pandas     version: {pd.__version__}\n"
    f"PyTorch version: {torch.__version__:.7}\n"
    )
logging.debug("\n\n"
    f"Plot figure size set to {mpl.rcParams['figure.figsize']}\n"
    f"Axes grid: {mpl.rcParams['axes.grid']}"
    )
#! Select GPU for usage. CPU versions ignores it.
#!! Learn to check if GPU is occupied or not.
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
if torch.cuda.is_available():
    #! With /device/GPU:1 the output was faster.
    #! need to research more why.
    # tf.config.experimental.set_visible_devices(
    #                         physical_devices[GPU], 'GPU')

    # if GPU == 1:
    # for device in physical_devices:
    #     tf.config.experimental.set_memory_growth(
    #                         device=device, enable=True)
    # tf.config.experimental.set_memory_growth(
    #                     device=physical_devices[GPU], enable=True)
    logging.info("GPU found") 
    
    logical_devices = "cuda"
    # pt.cuda.get_device_name()
    # pt.cuda.
    logging.info("GPU found") 
    # logging.debug(f"\nPhysical GPUs: {len(physical_devices)}"
    #               f"\nLogical GPUs: {len(logical_devices)}")
#! For numeric stability, set the default floating-point dtype to float64
# tf.keras.backend.set_floatx('float32')
# %%
# Train data
tic : float = time.perf_counter()
with open('Data/np-data/xy_train.npy', 'rb') as f:
    # x_train = Variable(torch.from_numpy(np.load(f)))
    x_train = torch.from_numpy(np.load(f)).to(logical_devices)
    y_train = torch.from_numpy(np.load(f)).to(logical_devices)
print(f"Train data loading took: {(time.perf_counter() - tic):.2f} seconds")

tic : float = time.perf_counter()
with open('Data/np-data/xyt_valid.npy', 'rb') as f:
    xt_valid = torch.from_numpy(np.load(f)).to(logical_devices)
    yt_valid = torch.from_numpy(np.load(f)).to(logical_devices)
print(f"\nValid data loading took: {(time.perf_counter() - tic):.2f} seconds")

# Test Battery data
tic : float = time.perf_counter()
with open('Data/np-data/xy_valid.npy', 'rb') as f:
    x_valid = torch.from_numpy(np.load(f)).to(logical_devices)
    y_valid = torch.from_numpy(np.load(f)).to(logical_devices)
print(f"\nTest battery data loading took: {(time.perf_counter() - tic):.2f} seconds")

# Test Cycles data
tic : float = time.perf_counter()
with open('Data/np-data/xy_testi.npy', 'rb') as f:
    x_testi = torch.from_numpy(np.load(f)).to(logical_devices)
    y_testi = torch.from_numpy(np.load(f)).to(logical_devices)
print(f"\nTest cycles data loading took: {(time.perf_counter() - tic):.2f} seconds")
# %%
class ManualModel(nn.Module):
    #TODO Declare vars

    #TODO: Add device
    def __init__(self, mFunc : Callable, layers : int = 1, neurons : int = 500,
                 dropout : float = 0.2, input_shape : tuple = (500,3),
                 batch : int = 1, device : str = 'cpu') -> None:
        super().__init__()
        # Check layers, neurons, dropout and batch are acceptable
        self.layers = 1 if layers == 0 else abs(layers)
        self.units : int = int(500/layers) if neurons == 0 else int(abs(neurons)/layers)
        dropout : float = float(dropout) if dropout >= 0 else float(abs(dropout))
        #? int(batch) if batch > 0 else ( int(abs(batch)) if batch != 0 else 1 )
        batch : int = int(abs(batch)) if batch != 0 else 1
        self.device = device

        # Define sequential model with an Input Layer
        self.tanh = nn.Tanh()
        self.model0 = mFunc(
                    input_size=input_shape[1], hidden_size=self.units,
                    num_layers=1, batch_first=True, device=device
                )
        self.model1 = mFunc(
                        input_size=self.units, hidden_size=self.units,
                        num_layers=1, batch_first=True, device=device
                )
        self.model2 = mFunc(
                        input_size=self.units, hidden_size=self.units,
                        num_layers=1, batch_first=True,device=device
                )
        self.dropout = nn.Dropout(p=dropout)
        
        self.sigmoind = nn.Sigmoid()
        self.output = nn.Linear(
                        in_features=self.units, out_features=1, device=device
                    )
        
    def forward(self,x):
        h_1 = Variable(torch.zeros(1, x.size(0), self.units)).to(self.device)
        c_1 = Variable(torch.zeros(1, x.size(0), self.units)).to(self.device)
        h_2 = Variable(torch.zeros(1, x.size(0), self.units)).to(self.device)
        c_2 = Variable(torch.zeros(1, x.size(0), self.units)).to(self.device)
        h_3 = Variable(torch.zeros(1, x.size(0), self.units)).to(self.device)
        c_3 = Variable(torch.zeros(1, x.size(0), self.units)).to(self.device)

        # Propagate input through LSTM
        _, (h_1, c_1) = self.model0(x, (h_1, c_1)) #LSTM with input, hidden, and internal state
        out = self.tanh(h_1)
        _, (h_2, c_2) = self.model1(out, (h_2, c_2)) #LSTM with input, hidden, and internal state
        out = self.tanh(h_2)
        _, (h_3, c_3) = self.model2(out, (h_3, c_3)) #LSTM with input, hidden, and internal state
        out = self.tanh(h_3)
        
        # Dropout
        out = self.dropout(out)
        
        #reshaping the data for Dense layer next
        out = self.output(h_3.view(-1, self.units))
        return self.sigmoind(out)#[:,0] # SoC output

def scheduler(epoch : int, lr : float, type : str = 'mix') -> float:
  """ Scheduler
  round(model.optimizer.lr.numpy(), 5)

  Args:
      epoch (int): [description]
      lr (float): [description]

  Returns:
      float: [description]
  """
  #! Think of the better sheduler
  if type == 'mix':
    if (epoch < 20):
        return lr
    else:
        # lr = tf_round(x=lr * tf.math.exp(-0.05), decimals=6)
        lr = lr * np.exp(-0.05)
        if lr >= 0.00005:
            return lr
        else:
            return  0.00005
  elif type == 'linear':
    return np.linspace(0.001, 0.0001, 100)[epoch]
  else:
    return 0.001

def get_learning_rate(epoch : int, iLr : float, type : str = 'mix') -> float:
  """_summary_

  Args:
      epoch (int): _description_
      iLr (float): _description_

  Returns:
      float: _description_
  """
  for i in range(0, epoch):
    iLr = scheduler(i, iLr, type)
  print(f'The Learning rate set to: {iLr}')
  return iLr

file_name : str = os.path.basename(__file__)[:-3]
model_name : str = 'ModelsPT-4'
####################! ADD model_name to path!!! ################################
model_loc : str = f'Modds/{model_name}/{nLayers}x{file_name}-({nNeurons})/{attempt}-{profile}/'
print(model_loc)
iEpoch : int = 0
firstLog : bool = True
iLr     : float = 0.001
prev_error : np.float32 = 1.0
try:
    iEpoch, prev_error  = Locate_Best_Epoch(f'{model_loc}history.csv', 'mae')
    iEpoch = int(iEpoch)
    model = ManualModel(mFunc=nn.LSTM, layers=3, neurons=131, dropout=0.2,
                    input_shape=x_train.shape[-2:], batch=1,
                    device=logical_devices)
    model.load_state_dict(torch.load(f'{model_loc}{iEpoch}'))
    iLr = get_learning_rate(iEpoch, iLr, 'linear')
    firstLog = False
    print(f"Model Identefied at {iEpoch} with {prev_error}. Continue training.")
except (OSError, TypeError) as identifier:
    print("Model Not Found, initiating new. {} \n".format(identifier))
    if type(x_train) == list:
        input_shape : tuple = x_train[0].shape[-2:]
    else:
        input_shape : tuple = x_train.shape[-2:]
    model = ManualModel(mFunc=nn.LSTM, layers=3, neurons=131, dropout=0.2,
                    input_shape=input_shape, batch=1, device=logical_devices)
    iLr = 0.001
    firstLog = True

prev_model = ManualModel(mFunc=nn.LSTM, layers=3, neurons=131, dropout=0.2,
                input_shape=x_train.shape[-2:], batch=1, device=logical_devices)
prev_model.load_state_dict(model.state_dict())
# %%
class RootMeanSquaredError(torchmetrics.Metric):
    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True
    def __init__(self) -> None:
        super().__init__()
        self.add_state("sum_squared_errors", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, target, preds):
        self.sum_squared_errors += torch.sum((preds - target) ** 2)
        self.n_observations += preds.numel()
    
    def compute(self):
        return torch.sqrt(self.sum_squared_errors / self.n_observations)

optimiser = RoAdam(params=model.parameters(), lr=1e-3,
                    betas=(0.9, 0.999, 0.999), eps=1e-07, amsgrad=False)
loss_fn = nn.L1Loss(
                reduction='none'
            )
# acc = torchmetrics.functional.accuracy(preds, target)
MAE = torchmetrics.MeanAbsoluteError().to(logical_devices)
RMSE = RootMeanSquaredError().to(logical_devices)
RSquare = torchmetrics.R2Score().to(logical_devices)
#! PT-1.12.1/lib/python3.10/site-packages/torch/nn/modules/loss.py:96:
#! UserWarning: Using a target size (torch.Size([1])) that is different to the
#! input size (torch.Size([1, 1])). This will likely lead to incorrect results 
#! due to broadcasting. Please ensure they have the same size.
def train_single_st(input : tuple[np.ndarray, np.ndarray],
                    metrics : list[torchmetrics.Metric]
                    ) -> float:
    # Compute prediction error
    logits = model(input[0])
    loss_value = loss_fn(target=input[1], input=logits)

    # Backpropagation
    optimiser.zero_grad()
    loss_value.backward()
    optimiser.step(loss_value[0][0])

    # Update metricks
    for metric in metrics:
        metric.update(target=input[1], preds=logits)

    return loss_value.item()

def valid_loop(dist_input  : tuple[np.ndarray, np.ndarray],
               verbose : int = 0) -> Variable:
    x, y = dist_input
    logits  : np.ndarray = torch.zeros(y.shape[0], 1).to(logical_devices)
    loss    : np.ndarray = torch.zeros(y.shape[0], 1).to(logical_devices)
    val_MAE     = torchmetrics.MeanAbsoluteError().to(logical_devices)
    val_RMSE    = RootMeanSquaredError().to(logical_devices)
    val_RSquare = torchmetrics.R2Score().to(logical_devices)
    
    # Debug verbose param
    if verbose == 1:
        rangeFunc : Callable = trange
    else:
        rangeFunc : Callable = range
    
    #! Prediction on this part can be paralylised across two GPUs. Like OpenMP
    model.eval()
    with torch.no_grad():
        for i in rangeFunc(y.shape[0]):
            logits[i] = model(x[i,:,:,:]).cpu()
            val_MAE.update(target=y[i],     preds=logits[i:i+1])
            val_RMSE.update(target=y[i],    preds=logits[i:i+1])
            val_RSquare.update(target=y[i], preds=logits[i:i+1])
            
            loss[i] = loss_fn(y[i], logits[i])
  
    mae      : float = val_MAE.compute()
    rmse     : float = val_RMSE.compute()
    r_Square : float = val_RSquare.compute()
    
    # Reset training metrics at the end of each epoch
    val_MAE.reset()
    val_RMSE.reset()
    val_RSquare.reset()

    return [loss, mae, rmse, r_Square, logits[:,0]]
# %%
if not os.path.exists(f'{model_loc}'):
    os.makedirs(f'{model_loc}')
if not os.path.exists(f'{model_loc}traiPlots'):
    os.mkdir(f'{model_loc}traiPlots')
if not os.path.exists(f'{model_loc}valdPlots'):
    os.mkdir(f'{model_loc}valdPlots')
if not os.path.exists(f'{model_loc}testPlots'):
    os.mkdir(f'{model_loc}testPlots')
if not os.path.exists(f'{model_loc}history.csv'):
    #! ADD EPOCH FOR FUTURE, THEN FIX THE BEST EPOCH
    print("History not created. Making")
    with open(f'{model_loc}history.csv', mode='w') as f:
        f.write('Epoch,loss,mae,rmse,rsquare,time(s),'
                'train_l,train_mae,train_rms,train_r_s,'
                'vall_l,val_mae,val_rms,val_r_s,val_t_s,'
                'test_l,tes_mae,tes_rms,tes_r_s,tes_t_s,learn_r\n')
if not os.path.exists(f'{model_loc}history-cycles.csv'):
    print("Cycle-History not created. Making")
    with open(f'{model_loc}history-cycles.csv', mode='w') as f:
        f.write('Epoch,Cycle,'
                'train_l,train_mae,train_rms,train_t_s,'
                'vall_l,val_mae,val_rms,val_t_s,'
                'learn_r\n')
if not os.path.exists(f'{model_loc}cycles-log'):
    os.mkdir(f'{model_loc}cycles-log')

#* Save the valid logits to separate files. Just annoying
# pd.DataFrame(y_train[:,0,0]).to_csv(f'{model_loc}y_train.csv', sep = ",",
#                                     na_rep = "", line_terminator = '\n')
# pd.DataFrame(yt_valid[:,0,0]).to_csv(f'{model_loc}yt_valid.csv', sep = ",",
#                                     na_rep = "", line_terminator = '\n')
# pd.DataFrame(y_valid[:,0,0]).to_csv(f'{model_loc}y_valid.csv', sep = ",",
#                                     na_rep = "", line_terminator = '\n')
# pd.DataFrame(y_testi[:,0,0]).to_csv(f'{model_loc}y_testi.csv', sep = ",",
#                                     na_rep = "", line_terminator = '\n')

n_attempts : int = 10
loss_value : float = 1.0
while iEpoch < mEpoch:
    iEpoch+=1
    model.train()
    pbar = tqdm(total=y_train.shape[0])
    tic : float = time.perf_counter()
    sh_i = np.arange(y_train.shape[0])
    np.random.shuffle(sh_i)
    # print(f'Commincing Epoch: {iEpoch}')
    optimiser = RoAdam(params=model.parameters(), lr=1e-3,
                    betas=(0.9, 0.999, 0.999), eps=1e-07, amsgrad=False)
    for i in sh_i[:]:
        loss_value = train_single_st((x_train[i,:,:,:],
                                      y_train[i,:]),
                                     [MAE,RMSE,RSquare]
                                )
        # Progress Bar
        pbar.update(1)
        pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
                            # f'loss: {loss_value:.4f} - '
                             f'mae: {MAE.compute():.4f} - '
                             f'rmse: {RMSE.compute():.4f} - '
                            #  f'rsquare: {RSquare.compute():.4f}'
                            )
    toc : float = time.perf_counter() - tic
    pbar.close()

    cLr = optimiser.param_groups[0]['lr']
    print(f'Epoch {iEpoch}/{mEpoch} :: '
                f'Elapsed Time: {toc} - '
                # f'loss: {loss_value[0]:.4f} - '
                f'mae: {MAE.compute():.4f} - '
                f'rmse: {RMSE.compute():.4f} - '
                f'rsquare: {RSquare.compute():.4f} - '
                f'Lear-Rate: {cLr} - '
            )
    
    #* Dealing with NaN state. Give few trials to see if model improves
    curr_error = MAE.compute().cpu().detach().numpy()
    print(f'The post optimiser error: {curr_error}', flush=True)
    if (np.isnan(loss_value) or curr_error > prev_error):
        print('->> NaN or High error model')
        i_attempts : int = 0
        firstFaltyLog : bool = True
        while i_attempts < n_attempts:
            print(f'->>> Attempt {i_attempts}')
            try:
                torch.save(obj=model.state_dict(),
                             f=f'{model_loc}{iEpoch}-fail-{i_attempts}')
            except OSError:
                os.remove(f'{model_loc}{iEpoch}-fail-{i_attempts}')
                torch.save(obj=model.state_dict(),
                             f=f'{model_loc}{iEpoch}-fail-{i_attempts}')

            model.load_state_dict(prev_model.state_dict())
            
            np.random.shuffle(sh_i)
            # pbar = tqdm(total=y_train.shape[0])

            # Reset every metric
            MAE.reset()
            RMSE.reset()
            RSquare.reset()
            optimiser = RoAdam(params=model.parameters(), lr=1e-3,
                    betas=(0.9, 0.999, 0.999), eps=1e-07, amsgrad=False)
            #! Potentially reset both curr and prev errors
            tic = time.perf_counter()
            model.train()
            for i in sh_i[::]:
                loss_value = train_single_st((x_train[i,:,:,:],
                                            y_train[i,:]),
                                            [MAE,RMSE,RSquare]
                                        )
                # Progress Bar
                # pbar.update(1)
                # pbar.set_description(f'Epoch {iEpoch}/{mEpoch} :: '
                #                     # f'loss: {(loss_value[0]):.4f} - '
                #                     )
            toc = time.perf_counter() - tic
            # pbar.close()
            TRAIN = valid_loop((xt_valid, yt_valid), verbose = debug)
            
            # Update learning rate
            iLr /= 2
            optimiser.param_groups[0]['lr'] = iLr

            if not os.path.exists(f'{model_loc}{iEpoch}-faulty-history.csv'):
                print(">> Creating faulty history")
                with open(f'{model_loc}{iEpoch}-faulty-history.csv', mode='w') as f:
                    f.write('Epoch,attempt,loss,mae,time(s),learn_r,'
                            'train_l,train_mae,train_rms,train_r_s\n'
                            )
            faulty_hist_df : pd.DataFrame = pd.read_csv(f'{model_loc}{iEpoch}-faulty-history.csv',
                                        index_col='Epoch')
            faulty_hist_df = faulty_hist_df.reset_index()
            # Log the faulty results
            faulty_hist_ser = pd.Series(data={
                    'Epoch'  : iEpoch,
                    'attempt': i_attempts,
                    'loss'   : loss_value,
                    'mae'    : MAE.compute().cpu().detach().numpy().item(),
                    'time(s)': toc,
                    'learn_r' : iLr,
                    'train_l' : torch.mean(TRAIN[0]).cpu().detach().numpy().item(),
                    'train_mae': TRAIN[1].cpu().detach().numpy().item(),
                    'train_rms': TRAIN[2].cpu().detach().numpy().item(),
                    'train_r_s': TRAIN[3].cpu().detach().numpy().item(),

                })
            # with open(f'{model_loc}{iEpoch}-faulty-history.csv',
            #             mode='a') as f:
            #     if(firstFaltyLog):
            #         faulty_hist_df.to_csv(f, index=False)
            #         firstFaltyLog = False
            #     else:
            #         faulty_hist_df.to_csv(f, index=False, header=False)
            if(len(faulty_hist_df[faulty_hist_df['Epoch']==iEpoch]) == 0):
                # faulty_hist_df = pd.concat([faulty_hist_df, faulty_hist_ser], ignore_index=True)
                hist_df = hist_df.append(faulty_hist_ser, ignore_index=True)
                # hist_df.loc[hist_df['Epoch']==iEpoch] = faulty_hist_ser
            else:
                faulty_hist_df.loc[len(faulty_hist_df)] = faulty_hist_ser
            faulty_hist_df.to_csv(f'{model_loc}history.csv', index=False, sep = ",", na_rep = "", line_terminator = '\n')

            curr_error = MAE.compute().cpu().detach().numpy()
            print(
                f'The post optimiser error: {curr_error}'
                f'with L-rate {optimiser.param_groups[0]["lr"]}'
                )
            if (not np.isnan(loss_value) and
                not curr_error > prev_error and
                not TRAIN[1] > 0.20 ):
                print(f'->>> Attempt {i_attempts} Passed')
                break
            else:
                i_attempts += 1
            # Collect garbage leftovers
            gc.collect()
        if (i_attempts == n_attempts):
            print('->> Model reached the optimum -- Breaking')
            break
        else:
            print('->> Model restored -- continue training')
            torch.save(obj=model.state_dict(), f=f'{model_loc}{iEpoch}')
            prev_model.load_state_dict(model.state_dict())
            prev_error = curr_error
    else:
        torch.save(obj=model.state_dict(), f=f'{model_loc}{iEpoch}')
        prev_model.load_state_dict(model.state_dict())
        prev_error = curr_error

    # Update learning rate
    iLr = scheduler(iEpoch, iLr, 'linear')
    optimiser.param_groups[0]['lr'] = iLr

    # Validating trained model 
    TRAIN = valid_loop((xt_valid, yt_valid), verbose = debug)
    RMS = (torch.sqrt(torch.square(
                yt_valid[:,0,0]-TRAIN[4])))
    predicting_plot(profile=profile, file_name=model_name,
                    model_loc=f'{model_loc}/traiPlots/',
                    model_type='LSTM valid',
                    iEpoch=f'tra-{iEpoch}',
                    Y=yt_valid[:,0].cpu(),
                    PRED=TRAIN[4].cpu(),
                    RMS=RMS.cpu(),
                    val_perf=[None, TRAIN[1],
                            TRAIN[2], TRAIN[3]],
                    TAIL=yt_valid.shape[0],
                    save_plot=True)
    print(f'Epoch {iEpoch}/{mEpoch} :: TRAIN :: '
            f'mae: {TRAIN[1]:.4f} - '
            f'rmse: {TRAIN[2]:.4f} - '
            f'rsquare: {TRAIN[3]:.4f} - '
            f'\n'
        )

    # Validating model 
    val_tic : float = time.perf_counter()
    PERF = valid_loop((x_valid, y_valid), verbose = debug)
    val_toc : float = time.perf_counter() - val_tic
    #! Verefy RMS shape
    #! if RMS.shape[0] == RMS.shape[1]
    RMS = (torch.sqrt(torch.square(
                y_valid[:,0,0]-PERF[4])))
    predicting_plot(profile=profile, file_name=model_name,
                    model_loc=f'{model_loc}/valdPlots/',
                    model_type='LSTM valid',
                    iEpoch=f'val-{iEpoch}',
                    Y=y_valid[:,0].cpu(),
                    PRED=PERF[4].cpu(),
                    RMS=RMS.cpu(),
                    val_perf=[None, PERF[1],
                            PERF[2], PERF[3]],
                    TAIL=y_valid.shape[0],
                    save_plot=True)
    print(f'Epoch {iEpoch}/{mEpoch} :: PERF :: '
            f'Elapsed Time: {val_toc} - '
            f'mae: {PERF[1]:.4f} - '
            f'rmse: {PERF[2]:.4f} - '
            f'rsquare: {PERF[3]:.4f} - '
            f'\n'
        )

    #! PErform testing and also save to log file
    # Testing model 
    mid_one = int(x_testi.shape[0]/2)#+350
    mid_two = int(x_testi.shape[0]/2)+400
    ts_tic : float = time.perf_counter()
    TEST1 = valid_loop((x_testi[:mid_one], y_testi[:mid_one]), verbose = debug)
    TEST2 = valid_loop((x_testi[mid_two:], y_testi[mid_two:]), verbose = debug)
    ts_toc : float = time.perf_counter() - ts_tic
    #! Verefy RMS shape
    RMS = (torch.sqrt(torch.square(
                y_testi[:mid_one,0,0]-TEST1[4])))
    #! If statement for string to change
    if profile == 'DST':
        save_title_type : str = 'LSTM Test on US06'
        save_file_name  : str = f'US06-{iEpoch}'
    else:
        save_title_type : str = 'LSTM Test on DST'
        save_file_name  : str = f'DST-{iEpoch}'

    predicting_plot(profile=profile, file_name=model_name,
                    model_loc=f'{model_loc}/testPlots/',
                    model_type=save_title_type,
                    iEpoch=save_file_name,
                    Y=y_testi[:mid_one,0].cpu(),
                    PRED=TEST1[4].cpu(),
                    RMS=RMS.cpu(),
                    val_perf=[None, TEST1[1],
                            TEST1[2], TEST1[3]],
                    TAIL=y_testi.shape[0],
                    save_plot=True)

    if profile == 'FUDS':
        save_title_type : str = 'LSTM Test on US06'
        save_file_name  : str = f'US06-{iEpoch}'
    else:
        save_title_type : str = 'LSTM Test on FUDS'
        save_file_name  : str = f'FUDS-{iEpoch}'
    #! Verefy RMS shape
    RMS = (torch.sqrt(torch.square(
                y_testi[mid_two:,0,0]-TEST2[4])))
    predicting_plot(profile=profile, file_name=model_name,
                    model_loc=f'{model_loc}/testPlots/',
                    model_type=save_title_type,
                    iEpoch=save_file_name,
                    Y=y_testi[mid_two:,0].cpu(),
                    PRED=TEST2[4].cpu(),
                    RMS=RMS.cpu(),
                    val_perf=[None, TEST2[1],
                            TEST2[2], TEST2[3]],
                    TAIL=y_testi.shape[0],
                    save_plot=True)
    print(f'Epoch {iEpoch}/{mEpoch} :: TEST :: '
            f'Elapsed Time: {ts_toc} - '
            f'mae: {np.mean(np.append(TEST1[1].cpu(), TEST2[1].cpu())):.4f} - '
            f'rmse: {np.mean(np.append(TEST1[2].cpu(), TEST2[2].cpu())):.4f} - '
            f'rsquare: {np.mean(np.append(TEST1[3].cpu(), TEST2[3].cpu())):.4f} - '
            f'\n'
        )

    hist_df : pd.DataFrame = pd.read_csv(f'{model_loc}history.csv',
                                            index_col='Epoch')
    hist_df = hist_df.reset_index()

    #! Rewrite as add, not a new, similar to the one I found on web with data analysis
    hist_ser = pd.Series(data={
            'Epoch'  : iEpoch,
            'loss'   : loss_value,
            'mae'    : MAE.compute().cpu().detach().numpy().item(),
            'rmse'   : RMSE.compute().cpu().detach().numpy().item(),
            'rsquare': RSquare.compute().cpu().detach().numpy().item(),
            'time(s)': toc,
            'train_l' : torch.mean(TRAIN[0]).cpu().detach().numpy().item(),
            'train_mae': TRAIN[1].cpu().detach().numpy().item(),
            'train_rms': TRAIN[2].cpu().detach().numpy().item(),
            'train_r_s': TRAIN[3].cpu().detach().numpy().item(),
            'vall_l' : torch.mean(PERF[0]).cpu().detach().numpy().item(),
            'val_mae': PERF[1].cpu().detach().numpy().item(),
            'val_rms': PERF[2].cpu().detach().numpy().item(),
            'val_r_s': PERF[3].cpu().detach().numpy().item(),
            'val_t_s': val_toc,
            'test_l' : np.mean(np.append(TEST1[0].cpu(), TEST2[0].cpu())),
            'tes_mae': np.mean(np.append(TEST1[1].cpu(), TEST2[1].cpu())),
            'tes_rms': np.mean(np.append(TEST1[2].cpu(), TEST2[2].cpu())),
            'tes_r_s': np.mean(np.append(TEST1[3].cpu(), TEST2[3].cpu())),
            'tes_t_s': ts_toc,
            'learn_r': iLr
        })
    if(len(hist_df[hist_df['Epoch']==iEpoch]) == 0):
        # hist_df = pd.concat([hist_df, hist_ser], ignore_index=True, axis=1)
        hist_df = hist_df.append(hist_ser, ignore_index=True)
        # hist_df.loc[hist_df['Epoch']==iEpoch] = hist_ser
    else:
        hist_df.loc[len(hist_df)] = hist_ser
    hist_df.to_csv(f'{model_loc}history.csv', index=False, sep = ",", na_rep = "", line_terminator = '\n')
    # print(hist_df)
    # print(hist_df.head())
    # Plot History for reference and overwrite if have to    
    history_plot(profile, model_name, model_loc, hist_df, save_plot=True,
                    plot_file_name=f'history-{profile}-train.svg')
    history_plot(profile, model_name, model_loc, hist_df, save_plot=True,
                metrics=['mae', 'val_mae',
                        'rmse', 'val_rms'],
                plot_file_name=f'history-{profile}-valid.svg')

    pd.DataFrame(TRAIN[4].cpu()).to_csv(f'{model_loc}{iEpoch}-train-logits.csv')
    pd.DataFrame(PERF[4].cpu()).to_csv(f'{model_loc}{iEpoch}-valid-logits.csv')
    pd.DataFrame(np.append(TEST1[4].cpu(), TEST2[4].cpu())
                ).to_csv(f'{model_loc}{iEpoch}-test--logits.csv')
                
    # Reset training metrics at the end of each epoch
    MAE.reset()
    RMSE.reset()
    RSquare.reset()
    del optimiser
    
    # Flush and clean
    print('\n', flush=True)

    # Collect garbage leftovers
    gc.collect()
# %%