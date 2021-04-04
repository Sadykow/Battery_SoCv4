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
(conda install -c conda-forge six==1.15.0 numpy==1.19.2 wheel==0.35 h5py==2.10.0 seaborn ipykernel pandas matplotlib scipy==1.5.2 xlrd) # Windows only
conda install numpy=1.19.2 wheel=0.35 h5py=2.10.0
conda install -c conda-forge numpy=1.19.2 wheel=0.35 h5py=2.10.0
python -m pip install -U --user keras_preprocessing --no-deps
conda install ipykernel
conda install pandas matplotlib scipy seaborn xlrd

cd tensorflow
python ./configure.py
bazel build //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/Temp/tensorflow_pkg
pip install ~/Temp/tensorflow_pkg/tensorflow-2.4.0-cp39-cp39-linux_x86_64.whl

bazel clean --expunge
sudo umount /tmp
'''
Supported instructions on my OS.
'''
/lib,/lib/x86_64-linux-gnu,/usr,/usr/lib/x86_64-linux-gnu/libfakeroot,/usr/local/cuda,/usr/local/cuda-11.1/targets/x86_64-linux/lib,/home/sadykov/OpenSource/TensorRT-7.2.2.3/lib,/home/sadykov/OpenSource/TensorRT-7.2.2.3/include

Supported instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA
--copt=-msse3 --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx --copt=-mavx2 --copt=-mfma

All supported by processor: --copt=-march=native
bazel build --config=cuda -c opt --copt=-march=native //tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tmp/tensorflow_pkg

'''

# Setting up TPU processor
'''
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
'''

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
