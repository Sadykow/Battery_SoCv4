#!/bin/bash -l
#PBS -N Ch-M40

#PBS -l walltime=00:20:00
#PBS -l mem=2GB
#PBS -l ncpus=2
#PBS -l ngpus=1
#PBS -l gputype=T4
module load tensorflow/2.3.1-fosscuda-2019b-python-3.7.4
python -m pip install --upgrade pip
python -m pip install --user matplotlib pandas numpy sklearn scipy tqdm openpyxl
python -m pip install ~/artifacts/tensorflow_addons-0.13.0-cp37-cp37m-linux_x86_64.whl

cd $PBS_O_WORKDIR


python Chemali2017.py -d False -e 100 -g 0 -p DST -l 3 -n 64 -a 1
