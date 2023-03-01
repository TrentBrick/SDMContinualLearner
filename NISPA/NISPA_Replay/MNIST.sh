#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research



for SEED in 0 19 42 2022 31415
do  
    CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py --experiment_name "MNIST_1000_${SEED}" \
    --dataset "mnist" --min_phases 3 --step_size_param 30 --model "mlp" --seed ${SEED} --prune_perc 90.0 --class_per_task 2  --learning_rate 0.1 --batch_size 256 --recovery_perc 0.5\
    --phase_epochs 5  --memo_size 1000 --replay_lambda 5 --optimizer "SGD" --grow_init "normal" 
done

exit 0