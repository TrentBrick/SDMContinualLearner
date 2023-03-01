#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research



#90 REG
for SEED in  0 19 42 2022 31415
do
    MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py --experiment_name "EFMNIST_${SEED}" --model mlp --rewire_algo full_random --seed ${SEED} --p_step_size cosine --step_size_param 30 --grow_init normal --reinit 1 --grow 1 --dataset emnist_fmnist --class_per_task 5 --prune_perc 80 --batch_size 512 --learning_rate 0.01 --recovery_perc 0.75 --phase_epochs 5
done
exit 0