# Battery_SoCv4

Repository is the 4th version of the Machine LEarning implementation of Battery
state of Charge prediction using Time Series methods.

This version relies on Tensorflow version 2.4 and numpy gets replaces with 
Tensor Numpy package in all implementations.

## Requirenments and environmental Software.
Package versions based on ./setup.py package requirenments.
'''
sudo mount --bind ~/Temp /tmp

conda create -n TF2.4-CPU python==3.9
conda activate TF2.4-CPU
conda install numpy=1.19.2 wheel=0.35 h5py=2.10.0
conda install -c conda-forge numpy=1.19.2 wheel=0.35 h5py=2.10.0
pip install -U --user keras_preprocessing --no-deps
conda install ipykernel
conda install pandas matplotlib scipy seaborn xlrd

cd tensorflow
./configure 
bazel build //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/Temp/tensorflow_pkg
pip install ~/Temp/tensorflow_pkg/tensorflow-2.4.0-cp39-cp39-linux_x86_64.whl

bazel clean --expunge
sudo umount /tmp
'''
