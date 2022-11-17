#!/bin/bash -l
script='Sadykov2022.py'
attempt=1
profiles='DST US06 FUDS'
steps=30

# A100
GPU='A100'
hours=200
index=1

for profile in $profiles
do
    layer=3
    neuron=131
    for attempt in {1..6}
    do
        # echo $profile $layer $neuron $attempt $steps
        bash gen_job_novel.bash $GPU $index $hours $script $profile $layer $neuron $attempt $steps > sub_job.pbs
        qsub sub_job.pbs
        rm sub_job.pbs
        ((index++))
    done
done