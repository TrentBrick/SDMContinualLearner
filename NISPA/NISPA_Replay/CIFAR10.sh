#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research

for SEED in 0 19 42 2022 31415
do  
    CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py --experiment_name "CIFAR_1000_${SEED}" \
    --dataset "cifar10" --step_size_param 40 --model "conv" --seed 0 --prune_perc 90 --class_per_task 2  --learning_rate 0.002 --batch_size 512 --recovery_perc 2.0 \
    --phase_epochs 5  --memo_size 1000 --replay_lambda 5 --min_phases 5 --optimizer "adam" --grow_init "normal" 
done



exit 0