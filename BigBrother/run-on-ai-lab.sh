#!/bin/bash

uv sync

#Maximum at ai-lab --mem=24G --cpus-per-task=15 --gres=gpu:8
#srun --mem=96G --cpus-per-task=60 --gres=gpu:4 --time=04:00:00
srun --mem=24G --cpus-per-task=15 --gres=gpu:1 --time=04:00:00 singularity exec --nv \
     -B ~/:/scratch \
     /ceph/container/tensorflow/tensorflow_24.09.sif \
     /bin/bash -lc ' \
       export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
       export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
       export TF_NUM_INTEROP_THREADS=2
       source /scratch/ESD7-Project/BigBrother/.venv/bin/activate && python /scratch/ESD7-Project/BigBrother/src/main.py
     '