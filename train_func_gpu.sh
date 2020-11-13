#!/bin/bash
#SBATCH -A venkvis
#SBATCH -p gpu
#SBATCH -n 64 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -o gpu_classif_output.%j # File to which STDOUT will be written
#SBATCH -e gpu_classif_error.%j # File to which STDERR will be written
#SBATCH --time=48:00:00
#SBATCH -J classif_train

echo "Job Started at `date`"

stdbuf -o0 -e0 command

/home/abills/julia-1.4.0/bin/julia train_func.jl

echo "Job Ended at `date`"