#!/bin/bash
#SBATCH --account=share-ie-itk
#SBATCH --partition=CPUQ
##SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=output_datasets_%j.txt
#SBATCH --job-name=make_datasets
#SBATCH --mail-user=trulsc00@gmail.com
#SBATCH --mail-type=ALL

set -euo pipefail

# Go to submission folder
cd ~/MODELS/Gear_cnn/Vessel-level-gear

# Load the software stack used to create the venv
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/11.8.0

# Activate the TensorFlow environment
source ~/venv/tf213/bin/activate

# Run Optuna tuning
python -u create_dataset_big.py