#!/bin/bash -l
script='Chemali2017.py'
attempt=1
# Job loops 30 hours
profiles='DST US06 FUDS'
# M40
GPU='M40'
hours=60
index=1
for profile in $profiles
do
    layer=1
    until [ $layer -gt 4 ]
    do
        for neuron in 32 65 131
        do
            # for attempt in {1..10}
            # do
            #     echo $profile $layer $neuron $attempt
            #     ((j++))
            # done            
            # python Chemali2017.py -d False -e 100 -g 0 -p $profile -l $layer -n $neuron -a $attempt
            bash gen_job.bash $GPU $index $hours $script $profile $layer $neuron $attempt > sub_job.pbs
            ## jobs/sub_job-$index.pbs
            # qsub sub_job.pbs
            # rm sub_job.pbs
            ((index++))
        done
        ((layer++))
    done
done

# T4
profiles='DST US06 FUDS'
GPU='T4'
hours=40
index=1
for profile in $profiles
do
    layer=1
    until [ $layer -gt 4 ]
    do
        for neuron in 262 524
        do
            bash gen_job.bash $GPU $index $hours $script $profile $layer $neuron $attempt > sub_job.pbs
            ## jobs/sub_job-$index.pbs
            # qsub sub_job.pbs
            # rm sub_job.pbs
            ((index++))
        done
        ((layer++))
    done
done