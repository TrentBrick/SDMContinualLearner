#90 REG
for SEED in 0
do
    MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py --experiment_name "split_CIFAR10_${SEED}" --model mlp --rewire_algo full_random --seed ${SEED} --p_step_size cosine --step_size_param 30 --grow_init normal --reinit 1 --grow 1 --dataset split_cifar10 --class_per_task 2 --prune_perc 40 --batch_size 512 --learning_rate 0.002 --recovery_perc 0.75 --phase_epochs 300 --multihead 0 --mask_outputs 0
done
exit 0