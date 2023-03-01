#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research


for SEED in 0 19 42 2022 31415
do
    #CIFAR10 
    CUBLAS_WORKSPACE_CONFIG=:16:8  python iBatchLearn.py --e_stopping_wait 3 --experiment_name CIFAR10_MAS_${SEED} --model_type conv --model_name ConvCustom --force_out_dim 0 --agent_type regularization --agent_name MAS --optimizer Adam --dataset CIFAR10 --first_split_size 2 --other_split_size 2 --batch_size 256 --lr 0.001 --schedule 50 --reg_coef 1 --seed ${SEED}
    
    #CIFAR100
    CUBLAS_WORKSPACE_CONFIG=:16:8  python iBatchLearn.py --optimizer Adam --e_stopping_wait 5 --experiment_name CIFAR100_MAS_${SEED} --model_type conv --model_name ConvCustom --force_out_dim 0 --agent_type regularization --agent_name MAS --optimizer Adam --dataset CIFAR100 --first_split_size 5 --other_split_size 5 --batch_size 32 --lr 0.001 --schedule 50 --reg_coef 1000 --seed ${SEED}
    
    #EFMNIST
    CUBLAS_WORKSPACE_CONFIG=:16:8  python iBatchLearn.py --optimizer Adam --e_stopping_wait 3 --experiment_name EFMNIST_MAS_${SEED} --model_type mlp --model_name MLPCustom --force_out_dim 0 --agent_type regularization --agent_name MAS --optimizer Adam --dataset TASK5 --first_split_size 10 --other_split_size 13 13 11 10 --batch_size 512 --lr 0.01 --schedule 20 --reg_coef 1000 --seed ${SEED}
done 
exit 0