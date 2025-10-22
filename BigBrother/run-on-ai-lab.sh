#!/bin/bash

uv sync

#Maximum at ai-lab --mem=24G --cpus-per-task=15 --gres=gpu:8
srun --mem=12G --cpus-per-task=8 --gres=gpu:4 singularity exec --nv \
     -B ~/ESD7-Project/BigBrother:/scratch/BigBrother \
     /ceph/container/tensorflow/tensorflow_24.09.sif \
     /bin/bash -c "source /scratch/BigBrother/.venv/bin/activate && python /scratch/BigBrother/src/main.py"
