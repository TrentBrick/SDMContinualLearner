#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research





for SEED in 0 19 42 2022 31415
do                      
    CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py --experiment_name "EFMNIST_5700_${SEED}" \
    --dataset "emnist_fmnist"  --model "mlp" --seed ${SEED} --prune_perc 70 --class_per_task 2  --learning_rate 0.05 --batch_size 128 --recovery_perc 2.0\
    --phase_epochs 5 --memo_size 5700--replay_lambda 1 --optimizer "SGD" --grow_init "normal"
done





exit 0