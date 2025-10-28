#!/bin/bash

uv sync

#Maximum at ai-lab --mem=24G --cpus-per-task=15 --gres=gpu:8
srun --mem=24G --cpus-per-task=15 --gres=gpu:4 singularity exec --nv \
     -B ~/:/scratch \
     /ceph/container/tensorflow/tensorflow_24.09.sif \
     /bin/bash -c "source /scratch/ESD7-Project/BigBrother/.venv/bin/activate && python /scratch/ESD7-Project/BigBrother/src/main.py"
