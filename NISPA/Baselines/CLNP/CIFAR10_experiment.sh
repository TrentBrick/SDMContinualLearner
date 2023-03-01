#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research


for SEED in  0
do
    MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python run_experiments.py --experiment_name CIFAR10_CLNP_${SEED} --dataset cifar10 --model conv --seed ${SEED} --class_per_task 2 --learning_rate 0.001 --batch_size 128 --l1_alpha 0.0001 --epochs 50 --m_perc 0.75 --theta_step 0.02
done 

exit 0