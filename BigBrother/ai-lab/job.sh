#!/bin/bash

#SBATCH --job-name=ai_template          # Name of your job
#SBATCH --output=ai_template.out        # Name of the output file
#SBATCH --error=ai_template.err         # Name of the error file
#SBATCH --mem=24G                       # Memory
#SBATCH --cpus-per-task=15              # CPUs per task
#SBATCH --gres=gpu:1                    # Allocated GPUs
#SBATCH --time=24:00:00                 # Maximum run time

singularity exec --nv ai_template.sif uv run ../src/main.py