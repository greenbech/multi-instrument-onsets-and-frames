module pruge
module load Python/3.8.6-GCCcore-10.2.0

source oaf/bin/activate
poetry run tensorboard --logdir=runs
