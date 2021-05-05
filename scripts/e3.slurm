#!/bin/bash
#SBATCH --job-name="use_openunmix"   # Sensible name for the job
#SBATCH --nodes=1             # Allocate 1 nodes for the job
#SBATCH -c28                  # Number of cores (can vary)
#SBATCH --time=00-02:30:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module load Python/3.8.6-GCCcore-10.2.0

# virtualenv openunmix
source openunmix/bin/activate
# pip install poetry
# poetry install

poetry run python e3.py
