#!/bin/bash
#
#SBATCH -A venkvis
#SBATCH -p cpu
#SBATCH -n 56 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem-per-cpu=2000 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o graph_preprop_out.%j # File to which STDOUT will be written
#SBATCH -e graph_preprop_err.%j # File to which STDERR will be written
#SBATCH --time=168:00:00
#SBATCH -J graph_preprop

/home/abills/julia-1.4.0/bin/julia graph_preprocessor_parallel.jl

echo "Job Ended at `date`"
