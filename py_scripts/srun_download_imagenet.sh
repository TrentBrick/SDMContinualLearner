#!/bin/bash
# USE SBATCH COMMAND TO CALL THIS!
#SBATCH -c 2                               # Request # cores
#SBATCH --mem=15G                          # Memory total in MB (for all cores)
#SBATCH -t 0-23:59                        # Runtime in D-HH:MM format

#SBATCH -p medium                         # Partition to run in
#SBATCH -N 1                            # Request one node (if you request more than one core with -c, also using
#SBATCH -o ../slurm_outputs/slurm-job_%j--array-ind_%a.out                 # File to which STDOUT + STDERR will be written, including job ID in filename

hostname
pwd

srun wget -q https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar