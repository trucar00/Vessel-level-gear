#!/bin/bash
#SBATCH --account=share-ie-itk
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=output_CNN_%j.txt
#SBATCH --job-name=CNN_gear_scale
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

echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Python: $(which python)"
python --version

# Locate pip-installed cuDNN 8.6
CUDNN_LIB=$(python - <<'PY'
import nvidia.cudnn
from pathlib import Path

print(Path(nvidia.cudnn.__file__).parent / "lib")
PY
)

export LD_LIBRARY_PATH="$CUDNN_LIB:${LD_LIBRARY_PATH:-}"

echo "cuDNN library directory: $CUDNN_LIB"

# Optional: avoid TensorFlow reserving all GPU memory immediately
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Disable automatic XLA compilation
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

nvidia-smi

# Verify TensorFlow and perform a real convolution test
python - <<'PY'
import sys
import tensorflow as tf

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Build information:", tf.sysconfig.get_build_info())

gpus = tf.config.list_physical_devices("GPU")
print("Visible GPUs:", gpus)

if not gpus:
    raise RuntimeError("TensorFlow cannot see the allocated GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

x = tf.random.normal((8, 120, 9))
layer = tf.keras.layers.Conv1D(32, 3, padding="same")

with tf.GradientTape() as tape:
    tape.watch(x)
    y = layer(x)
    loss = tf.reduce_mean(y ** 2)

gradients = tape.gradient(loss, layer.trainable_variables + [x])

print("Convolution output shape:", y.shape)
print("Backward pass succeeded:", all(g is not None for g in gradients))
print("Tensor device:", y.device)
PY

# Run Optuna tuning
python -u cnn_model_scale.py