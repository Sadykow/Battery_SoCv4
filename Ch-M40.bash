#!/bin/bash -l
#PBS -N Ch-M40

#PBS -l walltime=60:00:00
#PBS -l mem=2GB
#PBS -l ncpus=2
#PBS -l ngpus=1
#PBS -l gputype=M40
#PBS -J 1-40

# module load tensorflow/2.3.1-fosscuda-2019b-python-3.7.4
# python -m pip install --upgrade pip
# python -m pip install --user matplotlib pandas numpy sklearn scipy tqdm openpyxl
# python -m pip install ~/artifacts/tensorflow_addons-0.13.0-cp37-cp37m-linux_x86_64.whl

# cd /home/n9312706/MPhil/TF/Battery_SoCv4/

# python Chemali2017.py -d False -e 100 -g 0 -p FUDS -l 3 -n 131 -a 1
# Job loops 50 hours
# i = 0

##### Obtain Parameters from input.txt file using $PBS_ARRAY_INDEX as the line number #####
# M40 - 40
# T4 - 24

# for ((PBS_ARRAY_INDEX=1; PBS_ARRAY_INDEX<=24; PBS_ARRAY_INDEX++))
# do
parameters=`sed -n "${PBS_ARRAY_INDEX} p" input-M40.txt` 
paramArray=($parameters)

# echo ${paramArray[0]} ${paramArray[1]} ${paramArray[2]} ${paramArray[3]}
python Chemali2017.py -d False -e 100 -g 0 -p ${paramArray[0]} -l ${paramArray[1]} -n ${paramArray[2]} -a ${paramArray[3]}
# done
# 
# bash Model_1.bash $profile $layer $neuron $attempt
