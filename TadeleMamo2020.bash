#!/bin/bash -l
script='TadeleMamo2020.py'
attempt=1
profiles='DST US06 FUDS'

# A100
GPU='A100'
hours=200
index=1

for profile in $profiles
do
    layer=3
    neuron=131
    for attempt in {1..10}
    do
        # echo $profile $layer $neuron $attempt
        # python Chemali2017.py -d False -e 100 -g 0 -p $profile -l $layer -n $neuron -a $attempt
        bash gen_job.bash $GPU $index $hours $script $profile $layer $neuron $attempt > sub_job.pbs
        qsub sub_job.pbs
        rm sub_job.pbs
        ((index++))
    done
done
