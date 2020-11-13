#!/bin/bash
#SBATCH -A venkvis
#SBATCH -p cpu
#SBATCH -n 56 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem-per-cpu=2000 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o classif_output.%j # File to which STDOUT will be written
#SBATCH -e classif_error.%j # File to which STDERR will be written
#SBATCH --time=144:00:00
#SBATCH -J classif_train

echo "Job Started at `date`"

stdbuf -o0 -e0 command

export JULIA_NUM_THREADS=56
export OPENBLAS_NUM_THREADS=56

/home/abills/julia-1.4.0/bin/julia train_func.jl

echo "Job Ended at `date`"