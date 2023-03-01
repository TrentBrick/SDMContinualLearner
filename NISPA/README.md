# NISPA: Neuro-Inspired Stability-Plasticity Adaptation for Continual Learning in Sparse Networks


> **Abstract:** *The goal of  continual learning (CL)  is to learn different tasks over time. The main desiderata associated with CL are to maintain performance on older tasks, leverage the latter to improve learning of future tasks, and to introduce minimal overhead in the training process (for instance, to not require a growing model or retraining). We propose the Neuro-Inspired Stability-Plasticity Adaptation (NISPA) architecture that addresses these desiderata through a sparse neural network with fixed density. NISPA forms stable paths to preserve learned knowledge from older tasks. Also, NISPA uses connection rewiring to create new plastic paths that reuse existing knowledge on novel tasks. Our extensive evaluation on EMNIST, FashionMNIST, CIFAR10, and CIFAR100 datasets shows that NISPA significantly outperforms representative state-of-the-art continual learning baselines, and it uses up to ten times fewer learnable parameters compared to baselines. We also make the case that sparsity is an essential ingredient for continual learning.*

![NISPA](main_figure.png)

## Paper
https://arxiv.org/abs/2206.09117

## Code
This code was implemented using Python 3.8 (Anaconda) on Ubuntu 20.04.

Please use environment.yaml to create a conda environment.

EWC, SI, and MAS implementations are adapted from https://github.com/GT-RIPL/Continual-Learning-Benchmark

Please refer to https://github.com/aimagelab/mammoth for replay baselines presented in the paper. 

Each folder includes scripts to reproduce results presented in the paper. We used the following seeds: 0, 19, 42, 2022, 31415.

Once learning is completed, final test accuracy is written to results (xlsx file) and logs (txt file) directories.
