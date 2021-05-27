#!/bin/bash
#SBATCH --job-name="amt_e4"   # Sensible name for the job
#SBATCH --time=01-00:00:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

echo "Remember to run from project root folder!"
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module pruge
module load Python/3.8.6-GCCcore-10.2.0

source oaf/bin/activate
# pip install poetry
# poetry install

poetry run python -m experiments.e4
