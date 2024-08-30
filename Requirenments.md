## Requirenments and environmental Software.
Package versions based on ./setup.py package requirenments.
```
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

/lib,/lib/x86_64-linux-gnu,/usr,/usr/lib/x86_64-linux-gnu/libfakeroot,/usr/local/cuda,/usr/local/cuda-11.1/targets/x86_64-linux/lib,/home/sadykov/OpenSource/TensorRT-7.2.2.3/lib,/home/sadykov/OpenSource/TensorRT-7.2.2.3/include

Please make sure that
 -   PATH includes /opt/cuda12/bin
 -   LD_LIBRARY_PATH includes /opt/cuda12/lib64, or, add /opt/cuda12/lib64 to /etc/ld.so.conf and run ldconfig as root

/usr/local/cuda-11.7,/usr/local/cuda-11.7/include,/usr/include,/usr/include/x86_64-linux-gnu,/usr/lib/x86_64-linux-gnu,/home/user/TF/TensorRT/TensorRT-8.6.1.6/include,/home/user/TF/TensorRT/TensorRT-8.6.1.6/lib
/opt/cuda12,/opt/cuda12/include/,/home/user/TF/TensorRT/TensorRT-8.6.1.6/include,/home/user/TF/TensorRT/TensorRT-8.6.1.6/lib

bazel build //tensorflow/tools/pip_package:build_pip_package
( bazel build  --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" -s -c dbg //tensorflow/tools/pip_package:build_pip_package )
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/Temp/tensorflow_pkg
pip install ~/Temp/tensorflow_pkg/tensorflow-2.4.0-cp39-cp39-linux_x86_64.whl

bazel clean --expunge
sudo umount /tmp
```
Supported instructions on my OS.
```

Supported instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA
--copt=-msse3 --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx --copt=-mavx2 --copt=-mfma

All supported by processor: --copt=-march=native
bazel build --config=cuda -c opt --copt=-march=native //tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tmp/tensorflow_pkg

```