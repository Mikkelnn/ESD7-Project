THIS IS NOT WORKING, USE run-on-ai-lab.sh

# Setup singularity
export SINGULARITY_TMPDIR="$HOME/.singularity/tmp/"
export SINGULARITY_CACHEDIR="$HOME/.singularity/cache/"
mkdir -p $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR

# Build container 
srun singularity build --fakeroot esd7_project.sif esd7_project.def