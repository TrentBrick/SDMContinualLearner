#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research


for SEED in  0
do
    MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python run_experiments.py --experiment_name CIFAR100_CLNP_${SEED}  --dataset cifar100 --model conv --seed ${SEED} --class_per_task 5 --learning_rate 0.001 --batch_size 32 --l1_alpha 0.00001 --epochs 50 --m_perc 1 --theta_step 0.02
done 

exit 0