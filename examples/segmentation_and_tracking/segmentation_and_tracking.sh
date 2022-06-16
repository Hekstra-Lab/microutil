#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 15
#SBATCH -p seas_gpu_requeue,gpu_requeue
#SBATCH --mem-per-cpu 4G
#SBATCH -o logs/output_%a.txt
#SBATCH -e logs/errors_%a.txt
#SBATCH --gres gpu:1
#SBATCH --constraint v100

python segmentation_and_tracking_fluo.py /path/to/your/dataset.zarr "${SLURM_ARRAY_TASK_ID}"

# To run on a full dataset:
# sbatch --array=0-Number_of_scenes segmentation_and_tracking.sh
