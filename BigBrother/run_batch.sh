#!/bin/bash

#SBATCH --job-name=space-debris
#SBATCH --output=my_job.out
#SBATCH --error=my_job.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Run Python script in container
singularity exec --nv -B ~/:/scratch \
  /ceph/container/tensorflow/tensorflow_24.09.sif \
  /bin/bash -lc "export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}; \
  export TF_NUM_INTRAOP_THREADS=\${SLURM_CPUS_PER_TASK}; \
  export TF_NUM_INTEROP_THREADS=2; \
  source /scratch/ESD7-Project/BigBrother/.venv/bin/activate; \
  python /scratch/ESD7-Project/BigBrother/src/main.py"

# singularity exec --nv \ 
#     -B ~/:/scratch \
#     /ceph/container/tensorflow/tensorflow_24.09.sif \
#     /bin/bash -lc ' \
#       export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
#       export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
#       export TF_NUM_INTEROP_THREADS=2
#       source /scratch/ESD7-Project/BigBrother/.venv/bin/activate && python /scratch/ESD7-Project/BigBrother/src/main.py
#     '
