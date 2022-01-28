#!/bin/bash
#SBATCH --ntasks=1               # number of nodes
#SBATCH --cpus-per-task=18       # number of CPU cores per process
#SBATCH --output=job_%j.out     # file name for stdout/stderr
#SBATCH --error=job_%j.err
#SBATCH --mem=200G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=20:00:00         # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=test_job        # job name (default is the name of this file)

PYTHON=/home/chattbap/FederatedLearning_Framework/venv/bin/python
PROGRAM=/home/chattbap/FederatedLearning_Framework/extras/test_file_slurm.py

srun $PYTHON $PROGRAM
