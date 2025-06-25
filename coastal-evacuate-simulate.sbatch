#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 03:00:00
#SBATCH -p bii
#SBATCH -A nssac_covid19
#SBATCH --job-name=coastal-sim

module load miniforge
conda activate migration_env

python coastal-evacuate-simulation.py $1 $2 $3 $4 $5


