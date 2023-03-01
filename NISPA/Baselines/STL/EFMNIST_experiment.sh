#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research


for SEED in 0 19 42 2022 31415
do
    CUBLAS_WORKSPACE_CONFIG=:16:8 python single_task_experiment.py   --experiment_name "EFMNIST_STL_${SEED}"  --seed ${SEED}  --model "mlp" --dataset "emnist_fmnist" --prune_perc 80  --batch_size 128 --learning_rate 0.01 --epochs 20
    CUBLAS_WORKSPACE_CONFIG=:16:8 python single_task_experiment.py   --experiment_name "EFMNIST_STL_ISO_${SEED}"  --seed ${SEED}  --model "mlp" --dataset "emnist_fmnist" --prune_perc 96  --batch_size 128 --learning_rate 0.01 --epochs 20
done
exit 0