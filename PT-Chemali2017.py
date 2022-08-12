#!/usr/bin/python
# %% [markdown]
# # # 1
# # #
# # LSTM for SoC by Ephrem Chemali 2017
# PyTorch version for optimiser and custom training loop
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
from py_modules.utils import str2bool, Locate_Best_Epoch
from py_modules.plotting import predicting_plot, history_plot

from typing import Callable
if (sys.version_info[1] < 9):
    LIST = list
    from typing import List as list
    from typing import Tuple as tuple

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

opts = [('-d', 'False'), ('-e', '100'), ('-l', '3'), ('-n', '131'), ('-a', '11'),
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
        _, (h_1, c_1) = self.model0(x, (h_1, c_1)) #lstm with input, hidden, and internal state
        out = self.tanh(h_1)
        _, (h_2, c_2) = self.model1(out, (h_2, c_2)) #lstm with input, hidden, and internal state
        out = self.tanh(h_2)
        _, (h_3, c_3) = self.model2(out, (h_3, c_3)) #lstm with input, hidden, and internal state
        out = self.tanh(h_3)
        
        # Dropout
        out = self.dropout(out)
        
        #reshaping the data for Dense layer next
        out = self.output(h_3.view(-1, self.units))
        return self.sigmoind(out) # SoC output

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
model_name : str = 'ModelsPT-1'
####################! ADD model_name to path!!! ################################
model_loc : str = f'Modds/{model_name}/{nLayers}x{file_name}-({nNeurons})/{attempt}-{profile}/'
iEpoch = 0
firstLog : bool = True
iLr     : float = 0.001
prev_error : np.float32 = 1.0
try:
    iEpoch, prev_error  = Locate_Best_Epoch(f'{model_loc}history.csv', 'mae')
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
# train = DataLoader(dataGenerator.train_list[0],batch_size=1)
# test = DataLoader(dataGenerator.valid_list[0], batch_size=1)

# for X, y in train:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

# batch_size = 64

# # Create data loaders.
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break
# %%
# input_size - The number of expected features in the input x
# hidden_size - The number of features in the hidden state h
# num_layers - Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
# bias - If False, then the layer does not use bias weights b_ih and b_hh. Default: True
# batch_first - If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False
# dropout - If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
# bidirectional - If True, becomes a bidirectional LSTM. Default: False
# proj_size - If > 0, will use LSTM with projections of corresponding size. Default: 0

#? How does the hidden state gets splitted?
#! https://cnvrg.io/pytorch-lstm/
# model = nn.LSTM(input_size=3, hidden_size=131, num_layers=3, bias=True,
#                 batch_first=True, dropout=0.2,
#                 bidirectional=False, proj_size=0).to(logical_devices)
# model = nn.Sequential(OrderedDict([
#             ('LSTM1', nn.LSTM(input_size=3, hidden_size=43, num_layers=1, bias=True, batch_first=True, dropout=0.0)),
#             ('LSTM2', nn.LSTM(input_size=43, hidden_size=43, num_layers=1, bias=True, batch_first=True, dropout=0.0)),
#             ('LSTM3', nn.LSTM(input_size=43, hidden_size=43, num_layers=1, bias=True, batch_first=True, dropout=0.0)),
#             ('Dropout', nn.Dropout(p=0.2, inplace=False)),
#             ('Output', nn.Linear(in_features=43, out_features=1, bias=True,))
#         ])).to(logical_devices)
# model = LSTM_multi(1, 3, 43, 1, 1)
# model = LSTM_multi_output(1, 3, 43, 43, 43)
# model = MultiModel(mFunc=nn.LSTM, layers=3, neurons=131, dropout=0.2,
#                   input_shape=x_train.shape[2:], batch=1)
#! https://towardsdatascience.com/from-a-lstm-cell-to-a-multilayer-lstm-network-with-pytorch-2899eb5696f3
# %%
class RootMeanSquaredError(torchmetrics.Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("sum_squared_errors", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, target, preds):
        self.sum_squared_errors += torch.sum((preds - target) ** 2)
        self.n_observations += preds.numel()
    
    def compute(self):
        return torch.sqrt(self.sum_squared_errors / self.n_observations)

#! Epsilon taken from Tensorflow. PyTorch Default 1e-08
optimiser = torch.optim.Adam(params=model.parameters(), lr=1e-3,
                             betas=(0.9, 0.999), eps=1e-07, amsgrad=False)
loss_fn = nn.L1Loss(
                reduction='none'
            )
# acc = torchmetrics.functional.accuracy(preds, target)
MAE = torchmetrics.MeanAbsoluteError().to(logical_devices)
RMSE = RootMeanSquaredError().to(logical_devices)
RSquare = torchmetrics.R2Score().to(logical_devices)

def train_single_st(input : tuple[np.ndarray, np.ndarray],
                    metrics : list[torchmetrics.Metric]
                    ) -> float:
    # Compute prediction error
    logits = model(input[0])
    loss_value = loss_fn(target=input[1], input=logits)

    # Backpropagation
    optimiser.zero_grad()
    loss_value.backward()
    optimiser.step()

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

            # Log the faulty results
            faulty_hist_df = pd.DataFrame(data={
                    'Epoch'  : [iEpoch],
                    'attempt': [i_attempts],
                    'loss'   : [np.array(loss_value)],
                    'mae'    : [np.array(MAE.compute().cpu())],
                    'time(s)': [np.array(toc)],
                    'learning_rate' : [np.array(iLr)],
                    'train_l' : torch.mean(TRAIN[0]).cpu(),
                    'train_mae': np.array(TRAIN[1].cpu()),
                    'train_rms': np.array(TRAIN[2].cpu()),
                    'train_r_s': np.array(TRAIN[3].cpu()),
                })
            with open(f'{model_loc}{iEpoch}-faulty-history.csv',
                        mode='a') as f:
                if(firstFaltyLog):
                    faulty_hist_df.to_csv(f, index=False)
                    firstFaltyLog = False
                else:
                    faulty_hist_df.to_csv(f, index=False, header=False)
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
        save_title_type : str = 'GRU Test on US06'
        save_file_name  : str = f'US06-{iEpoch}'
    else:
        save_title_type : str = 'GRU Test on DST'
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
        save_title_type : str = 'GRU Test on US06'
        save_file_name  : str = f'US06-{iEpoch}'
    else:
        save_title_type : str = 'GRU Test on FUDS'
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
            'loss'   : np.array(loss_value),
            'mae'    : np.array(MAE.compute().cpu()),
            'rmse'   : np.array(RMSE.compute().cpu()),
            'rsquare': np.array(RSquare.compute().cpu()),
            'time(s)': toc,
            'train_l' : torch.mean(TRAIN[0]).cpu(),
            'train_mae': np.array(TRAIN[1].cpu()),
            'train_rms': np.array(TRAIN[2].cpu()),
            'train_r_s': np.array(TRAIN[3].cpu()),
            'vall_l' : torch.mean(PERF[0]).cpu(),
            'val_mae': np.array(PERF[1].cpu()),
            'val_rms': np.array(PERF[2].cpu()),
            'val_r_s': np.array(PERF[3].cpu()),
            'val_t_s': val_toc,
            'test_l' : np.mean(np.append(TEST1[0].cpu(), TEST2[0].cpu())),
            'tes_mae': np.mean(np.append(TEST1[1].cpu(), TEST2[1].cpu())),
            'tes_rms': np.mean(np.append(TEST1[2].cpu(), TEST2[2].cpu())),
            'tes_r_s': np.mean(np.append(TEST1[3].cpu(), TEST2[3].cpu())),
            'tes_t_s': ts_toc,
            'learn_r': np.array(iLr)
        })
    if(len(hist_df[hist_df['Epoch']==iEpoch]) == 0):
        hist_df = pd.concat([hist_df, hist_ser], ignore_index=True)
        # hist_df = hist_df.append(hist_ser, ignore_index=True)
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

    # Flush and clean
    print('\n', flush=True)
# %%
# X = dataGenerator.train_list[0].loc[
#     0:499,['Voltage(V)', 'Current(A)', 'Temperature (C)_1']
#     ].to_numpy()
# y = dataGenerator.train_list[0].loc[
#     499,['Charge_Capacity(Ah)']
#     ].to_numpy()
"""
train = DataLoader((X,y), batch_size=1)

for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(logical_devices), y.to(logical_devices)
    train_single_st((X, y))
    if batch % 100 == 0:
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#! Testing loop
model.eval()
test_loss, correct = 0, 0
results = np.zeros(shape=(xt_valid.shape[0]))
with torch.no_grad():
    #x_train, y_train = x_train.to(logical_devices), y_train.to(logical_devices)
    for i in trange(xt_valid.shape[0]):
        results[i] = model(xt_valid[i,:,:,:])

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
test_loss /= num_batches
correct /= size
print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#! Save model
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

#! Load model
model = nn.LSTM(input_size=3, hidden_size=131, num_layers=3, bias=True,
                batch_first=True, dropout=0.2, bidirectional=False, proj_size=0)
model.load_state_dict(torch.load("model.pth"))

# %%

class LSTM_multi_output(nn.Module):
    def __init__(self, step_size, input_dimensions, first_hidden_size, second_hidden_size, third_hidden_size):
      super(LSTM_multi_output, self).__init__()
      self.first_hidden_size = first_hidden_size
      self.second_hidden_size = second_hidden_size
      self.third_hidden_size = third_hidden_size
      self.step_size = step_size
      self.input_dimensions = input_dimensions
      self.first_layer = nn.LSTM(input_size = self.input_dimensions,
                                 hidden_size = self.first_hidden_size, 
                                 num_layers = 1, batch_first = True)
      self.second_layer = nn.LSTM(input_size = self.first_hidden_size,
                                 hidden_size = self.second_hidden_size, 
                                 num_layers = 1, batch_first = True)
      self.third_layer = nn.LSTM(input_size = self.second_hidden_size,
                                 hidden_size = self.third_hidden_size,
                                 num_layers = 1, batch_first = True)
      self.fc_layer = nn.Linear(self.step_size*self.third_hidden_size, 4)
    def forward(self, x):
      seq_len, batch_size, _, _ = x.size()
      h_1 = torch.zeros(1, batch_size, self.first_hidden_size)
      c_1 = torch.zeros(1, batch_size, self.first_hidden_size)
      hidden_1 = (h_1, c_1)
      lstm_out, hidden_1 = self.first_layer(x, hidden_1)
      h_2 = torch.zeros(1, batch_size, self.second_hidden_size)
      c_2 = torch.zeros(1, batch_size, self.second_hidden_size)
      hidden_2 = (h_2, c_2)
      lstm_out, hidden_2 = self.second_layer(lstm_out, hidden_2)
      h_3 = torch.zeros(1, batch_size, self.third_hidden_size)
      c_3 = torch.zeros(1, batch_size, self.third_hidden_size)
      hidden_3 = (h_3, c_3)
      lstm_out, hidden_3 = self.third_layer(lstm_out, hidden_3)
      x = lstm_out.contiguous().view(batch_size,-1)
      return self.fc_layer(x)

class LSTM_multi(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size,
                 num_layers, seq_length):
        super(LSTM_multi, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True) #lstm
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True) #lstm
        self.lstm3 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True) #lstm
        self.fc = nn.Linear(hidden_size, num_classes) #fully connected last layer

        # self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoind = nn.Sigmoid()
    
    def forward(self,x):
        samples, batch_size, seq_len, num_classes = x.size()
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        h_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        h_3 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_3 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state

        # Propagate input through LSTM
        _, (h_1, c_1) = self.lstm1(x[:,0,:,:],   (h_1, c_1)) #lstm with input, hidden, and internal state
        out = self.tanh(h_1)
        _, (h_2, c_2) = self.lstm2(out, (h_2, c_2)) #lstm with input, hidden, and internal state
        out = self.tanh(h_2)
        _, (h_3, c_3) = self.lstm3(out, (h_3, c_3)) #lstm with input, hidden, and internal state
        out = self.tanh(h_3)

        h_3 = h_3.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.fc(h_3) #first Dense
        return self.sigmoind(out) # SoC output

class MultiModel(nn.Module):
    #TODO Declare vars

    #TODO: Add device
    def __init__(self, mFunc : Callable, layers : int = 1, neurons : int = 500,
                 dropout : float = 0.2, input_shape : tuple = (500,3),
                 batch : int = 1) -> None:
        super().__init__()
        # Check layers, neurons, dropout and batch are acceptable
        self.layers = 1 if layers == 0 else abs(layers)
        self.units : int = int(500/layers) if neurons == 0 else int(abs(neurons)/layers)
        dropout : float = float(dropout) if dropout >= 0 else float(abs(dropout))
        #? int(batch) if batch > 0 else ( int(abs(batch)) if batch != 0 else 1 )
        batch : int = int(abs(batch)) if batch != 0 else 1

        # Define sequential model with an Input Layer
        self.model = []        
        self.model.append(mFunc(
                    input_size=input_shape[1], hidden_size=self.units,
                    num_layers=1, batch_first=True
                ))
        # Fill the layer content
        if(self.layers > 1): #* Middle connection layers
            for _ in range(layers-1):
                self.model.append(mFunc(
                        input_size=self.units, hidden_size=self.units,
                        num_layers=1, batch_first=True
                    ))
        if(self.layers > 0):  #* Last no-connection layer
            self.model.append(nn.Linear(
                        in_features=self.units, out_features=1
                    ))
        else:
            print("Unhaldeled exeption with Layers")
            raise ZeroDivisionError
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoind = nn.Sigmoid()
    
    def forward(self, input):
        h_c = []
        for i in range(self.layers):
            h_c.append(
                (Variable(torch.zeros(1, input.size(0), self.units)), #hidden state
                Variable(torch.zeros(1, input.size(0), self.units))) #internal state
            )
        
        _, h_c[0] = self.model[0](input[:,:,:], h_c[0])
        out = self.tanh(h_c[0][0])
        for i in range(1, self.layers):
            # Propagate input through LSTM
            _, h_c[i] = self.model[i](out, h_c[i])
            out = self.tanh(h_c[i][0])

        h_c[-1][0] = h_c[-1][0].view(-1, self.units) #reshaping the data for Dense layer next
        out = self.model[-1](h_c[-1][0]) #first Dense
        return self.sigmoind(out) # SoC output
"""
