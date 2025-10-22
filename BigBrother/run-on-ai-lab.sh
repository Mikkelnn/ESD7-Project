#!/bin/bash

uv sync

srun --gres=gpu:1 singularity exec --nv \
     -B ~/ESD7-Project/BigBrother:/scratch/BigBrother \
     /ceph/container/tensorflow/tensorflow_24.09.sif \
     /bin/bash -c "source /scratch/BigBrother/.venv/bin/activate && python /scratch/BigBrother/src/main.py"
