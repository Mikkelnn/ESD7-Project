#!/bin/bash

#SBATCH --job-name=esd7_project          # Name of your job
#SBATCH --output=esd7_project.out        # Name of the output file
#SBATCH --error=esd7_project.err         # Name of the error file
#SBATCH --mem=24G                        # Memory
#SBATCH --cpus-per-task=15               # CPUs per task
#SBATCH --gres=gpu:1                     # Allocated GPUs
#SBATCH --time=11:30:00                  # Maximum run time

srun singularity build --fakeroot esd7_project.sif esd7_project.def # Build container

singularity exec --nv esd7_project.sif uv run ../src/main.py # Run simulations
