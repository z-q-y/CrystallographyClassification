#!/bin/bash
#
#SBATCH -A venkvis
#SBATCH -p cpu
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem-per-cpu=8000 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output.%j # File to which STDOUT will be written
#SBATCH -e error.%j # File to which STDERR will be written
#SBATCH --time=0-08:00
#SBATCH -J conv

/home/abills/julia-1.4.0/bin/julia graph_reader_test.jl

echo "Job Ended at `date`"