#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research


for SEED in 0 19 42 2022 31415
do
    CUBLAS_WORKSPACE_CONFIG=:16:8 python single_task_experiment.py   --experiment_name "CIFAR10_STL_${SEED}"   --seed ${SEED} --model "conv" --dataset "cifar10" --class_per_task 2 --prune_perc 90 --batch_size 256 --learning_rate 0.001 --epochs 40
    CUBLAS_WORKSPACE_CONFIG=:16:8 python single_task_experiment.py   --experiment_name "CIFAR10_STL_ISO_${SEED}"   --seed ${SEED} --model "conv" --dataset "cifar10" --class_per_task 2 --prune_perc 98 --batch_size 256 --learning_rate 0.001 --epochs 40
done
exit 0