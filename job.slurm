#!/bin/bash
#SBATCH --account=share-ie-itk
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=8     # 2 CPU cores
#SBATCH --mem=64G             # GB memory
#SBATCH --output=output_LSTM_%j.txt    # Log file
#SBATCH --job-name=CNN_gear_tune_test
#SBATCH --mail-user=trulsc00@gmail.com
#SBATCH --mail-type=ALL

# Go to submission folder
cd ~/MODELS/Gear_cnn/Vessel-level-gear

# Load Python
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Activate venv
source ~/venv/py311/bin/activate # VENV FOR GPU

# Run your script
python -u cnn_tune.py