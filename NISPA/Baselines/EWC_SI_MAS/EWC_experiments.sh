#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research


for SEED in 0 19 42 2022 31415
do 
    #CIFAR10 
    CUBLAS_WORKSPACE_CONFIG=:16:8  python iBatchLearn.py --e_stopping_wait 3 --experiment_name EWC_CIFAR10_${SEED} --model_type conv --model_name ConvCustom --force_out_dim 0 --agent_type regularization --agent_name EWC --optimizer Adam --dataset CIFAR10 --first_split_size 2 --other_split_size 2 --batch_size 128 --lr 0.001 --schedule 50 --reg_coef 5000 --seed ${SEED} --n_samples 2000
    
    #CIFAR100
    CUBLAS_WORKSPACE_CONFIG=:16:8  python iBatchLearn.py --e_stopping_wait 3 --optimizer Adam --experiment_name EWC_CIFAR100_${SEED} --model_type conv --model_name ConvCustom --force_out_dim 0 --agent_type regularization --agent_name EWC --optimizer Adam --dataset CIFAR100 --first_split_size 5 --other_split_size 5 --batch_size 32 --lr 0.001 --schedule 50 --reg_coef 20000 --seed ${SEED} --n_samples 1000
    
    #EFMNIST
    CUBLAS_WORKSPACE_CONFIG=:16:8  python iBatchLearn.py --e_stopping_wait 3 --optimizer Adam --experiment_name EWC_EFMNIST_${SEED} --model_type mlp --model_name MLPCustom --force_out_dim 0 --agent_type regularization --agent_name EWC --optimizer Adam --dataset TASK5 --first_split_size 10 --other_split_size 13 13 11 10 --batch_size 256 --lr 0.01 --schedule 20 --reg_coef 10000 --seed ${SEED} --n_samples 1000
done 
exit 0