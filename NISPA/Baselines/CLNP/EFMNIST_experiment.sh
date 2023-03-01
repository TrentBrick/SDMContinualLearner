#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research


for SEED in  0
do
    MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python run_experiments.py --experiment_name EFMNIST_CLNP_${SEED} --dataset emnist_fmnist --model mlp --seed ${SEED} --class_per_task 2 --learning_rate 0.01 --batch_size 512 --l1_alpha 0.00001 --epochs 20 --m_perc 0.5 --theta_step 0.02
done 

exit 0