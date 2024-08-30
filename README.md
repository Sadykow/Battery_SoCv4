# Battery_SoCv4

Repository is the 4th version of the Machine Learning implementation of Battery
state of Charge prediction using Time Series methods.

This version relies on Tensorflow version 2.4 and numpy gets replaces with 
Tensor Numpy package in all implementations.

Refer to [Requirenments.md](Requirenments.md) for installation details.

## References
The research has been split into 2 parts (formally 3): Investigation and Development.
The first part involved researching already published models, implement and adapt to unified methodology to fit the application of Electric Vehicles and make comparable judgement on the performance.
The `*.bash` scripts served as HPC setups for multiple models jobs creation.
The array of jobs hasn't been supported, so the loop submission was the only feasible solution.
Following references were implemented with corresponding Python script files.

---
* E. Chemali, P. J. Kollmeyer, M. Preindl, and A. Emadi, “State-of-charge estimation of Li-ion batteries using deep neural networks: A machine learning approach,” Journal of Power Sources, vol. 400, pp. 242–255, Oct. 2018, doi: 10.1016/j.jpowsour.2018.06.104.

Chemali et al was the foundation of this work with LSTM cells.
His approach as the simplest and most direct served as starting point for other more complex algorithms. Following files outline his implementation:

[Chemali***.py](Chemali2017.py) - Outlining single or multiwork implementation, including multifeature implementation, or more advanced optimisers.
His works was also tested against the [GRU based models](Ghemali2017.py). 
There was an attempt of using PyTorch version for performance measurements and custom optimiser tests.

---
* Song, Y.; Li, L.; Peng, Y.; Liu, D. Lithium-Ion Battery Remaining Useful Life Prediction Based on GRU-RNN. In Proceedings of
the 2018 12th International Conference on Reliability, Maintainability, and Safety (ICRMS), Shanghai, China, 17–19 October 2018;
pp. 317–322.

Multilayer implementations and assessment of the GRU against LSTM, with less gates and potentially faster workout.
Thanks to [Song et al](YuchenSong2018.py) input, the researched explored the gates of the ML cell implementation.
The GRU was meant to be used for the novel method, but it already was complex and any further assessment would open more questions for investigation.

---
* C. Li, F. Xiao, and Y. Fan, “An Approach to State of Charge Estimation of Lithium-Ion Batteries Based on Recurrent Neural Networks with Gated Recurrent Unit,” Energies, vol. 12, no. 9, p. 1592, Jan. 2019, doi: 10.3390/en12091592.

[BinXiao**.py](BinXiao2020.py) - The GRU models using stateful data handling and batching mechanism. The implementation found its way into the novel approach, where the input windows had been modified for Autoregressive purposes.

---
* T. Mamo and F. -K. Wang, "Long Short-Term Memory With Attention Mechanism for State of Charge Estimation of Lithium-Ion Batteries," in IEEE Access, vol. 8, pp. 94140-94151, 2020, doi: 10.1109/ACCESS.2020.2995656.

[Mamo et al](TadeleMamo2020.py) work was an example of modifying the model with additional logic, in this case - the Attention layer.
At the time of the research, no Attention was implemented in the framework, therefore in was manually created relying [on other source (gentaiscool)](https://github.com/gentaiscool/lstm-attention).

This worked helped the implementation of Custom models, making its way to novel implementation.

---
* Jiao, M.; Wang, D.; Qiu, J. A GRU-RNN based momentum optimized algorithm for SOC estimation. J. Power Sources 2020,
459, 228051

[Meng Jiaos'](MengJiao2020.py) work was the first attempt to compare the efficiency of the optimisers.
Adding a momentum to standard Gradient Descent optimers allowed to track the progression of the optimisers complexity mathematically, which found its' way in the first article.

---
* Javid, Gelareh & Basset, Michel & Ould Abdeslam, Djaffar. (2020). Adaptive Online Gated Recurrent Unit for Lithium-Ion Battery SOC Estimation. 10.1109/IECON43393.2020.9254506. 

[Gelareh Javid](GelarehJavid2020.py) used the Robust Adam optimiser, which is not available within TensorFlow due to hardened compiled library for Adam.
The [PyTorch version](py_modules/PT_RobustAdam.py) was created to get the understanding of the optmisier, which uses runtime variable as a hyperparameter for the optimisation (change of the loss values) and then custom recreated by inheriting Optimser class, [RoAdam](py_modules/RobustAdam.py).

---
* Zhang, W.; Li, X.; Li, X. Deep Learning-Based Prognostic Approach for Lithium-ion Batteries with Adaptive Time-Series
Prediction and On-Line Validation. Measurement 2020, 164, 108052.

[Zhang et al](WeiZhang2020.py) used the online implementation for validation.
It meant to be example of how to use SoC estimation in the live driving, with actual sensory data.
Similar techniques was meant to be applied with battery cycling machine for either single cell or a car accumulator, and live driving with Wireless telemetry to the remote station.
Later it meant to be applied a TPU processors with Artemis controllers.


## Publications
* M. Sadykov, S. Haines, M. Broadmeadow, G. Walker, and D. W. Holmes, “Practical evaluation of Lithium-Ion battery State of Charge estimation using Time-Series Machine Learning for Electric Vehicles,” Energies, Dec. 2022, Accessed: Dec. 23, 2022. [Online](https://susy.mdpi.com/user/manuscripts/review_info/b800911c303f2ea65f8f8b17aba7dabb).

The First publication is focused on the Methodology and the comparison of the 5 unique models, identify the flaws and study the implementations.

To speed up the process of creating each model for 3 datasets 10 times, there were several additional algorithms, which were created to speed up the process and allowed the repentance of the work without setting the strict seed value:
* Custom training/testing functions with `@tf.function` decarator.
* Optimiser hyperparameter reduction is the MAE increased after single training epoch. The Learning rate reduction based on a scheduler.
* The model will be rolled back and the reduced learning rate value will be used.
After set amount of attempts, if the error does not reduce, the model considered at its' minimum.
* Multiple plots, CSV data creation, weights and model clones storage, and ClickHouse database results exports were used to preserve the raw results.
The post-processing has been done after the training and away from HPC machines using synchronised computation on 2 GPUs and Single CPU unit.

---
* M. Sadykov, S. Haines, G. Walker, and D. W. Holmes, “Feed-forward State of Charge estimation of LiFePO4 batteries using time-series machine learning prediction with autoregressive models,” Journal of Energy Storage, vol. 100, p. 113516, 2024, doi: https://doi.org/10.1016/j.est.2024.113516.

The novel implementation within [Sadykov*.py](Sadykov2022.py) python scripts. 
The idea was to use Autoregression method of hadeling the training, but modefing the training process with Feed-Forward results input to introduce potential inaccuracies as part of the system.
The Cell structure of GRU and Optimers of the RoAdam were meant to be used to make up for the potential time loss in training process.
However, at this stage, given no other published methods of similar technique at the time, it was decided to keep the implementation as close to general as possible.

## Additional scripts
Multiple other scripts were created to support the thesis and publications with plots, flops counters.

The data taken from CALCE researched were prepocessed with [DataGenerator](extractor/DataGenerator.py) and [WindowGenerator](extractor/WindowGenerator.py) extractors.
This separation explored multiple ways of speeding up and handeling the inputs, in particular multithreaded and `BatchedDataset` creators.

ClickHouse data management with cloud server setup on RaspberryPi was one of the more complex implementations, given that the over hundreds of models had to be preprocessed and properly stored for the publication purposes.
Fortunately, the relationships were simple to work with.
The left over credentials carry no value, since the Raspbeery Pi server existed only in local network and currently shutdown.

# Setting up TPU processor
```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update

    # Select only 1 of following
sudo apt-get install libedgetpu1-std    # Standard frequency
sudo apt-get install libedgetpu1-max    # Max frequency 

/etc/udev/rules.d/99-edgetpu-accelerator.rules
SUBSYSTEM=="usb",ATTRS{idVendor}=="1a6e",GROUP="plugdev"
SUBSYSTEM=="usb",ATTRS{idVendor}=="18d1",GROUP="plugdev"

Conda Raspberry 4 requirenments
PackageNotFoundError: Package not found: '' Packages missing in current linux-armv7l channels: 
  - numpy ==1.19.2
  - h5py ==2.10.0
python -m pip install six==1.15.0 numpy==1.19.2 wheel==0.35 h5py==2.10.0 seaborn ipykernel pandas matplotlib scipy xlrd
```

# Code utilises C libraries in the c_modules.
Setup with following command inside directory. Not working yet. Something to do
with strings in C and Python3. Utf8 and ASCI does not make easy comparison.
Try encode, perhaps.
```
python setup.py build
```
# Code utilises Cython libraries in the cy_modules.
Setup with following command inside directory.
```
python setup.py build_ext --inplace
```
