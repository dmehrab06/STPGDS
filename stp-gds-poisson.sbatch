#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 05:00:00
#SBATCH -p bii
#SBATCH -A nssac_covid19
#SBATCH --job-name=active-N

module load miniforge
conda activate migration_env

python stp-gds-poisson.py $1 $2 $3


