#!/bin/bash
#
#SBATCH -A venkvis
#SBATCH -p cpu
#SBATCH -n 112 # Number of cores
#SBATCH -N 2 # Ensure that all cores are on one machine
#SBATCH --mem-per-cpu=2000 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o graph_prep_output_N%D_C%C.%j # File to which STDOUT will be written
#SBATCH -e graph_prep_error_N%D_C%C.%j # File to which STDERR will be written
#SBATCH --time=168:00:00
#SBATCH -J cif_preprocess

echo "Job Started at `date`"

/home/abills/julia-1.4.0/bin/julia graph_preprocessor.jl

echo "Job Ended at `date`"