#!/bin/bash
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --gpus=1                        
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --job-name=rvae
#SBATCH --time=12:00:00
#SBATCH --output=/home/knum/run_logs/outputs/%x-%j.out
#SBATCH --error=/home/knum/run_logs/errors/%x-%j.error
#SBATCH --export=ALL

# Append CUDA path to the existing PATH
export PATH="/usr/local/cuda/bin:${PATH}"

# Define general user specific constants
# --- USER SPECIFIC CONSTANTS START ---
ENV_PATH="/home/knum/projekty/envs/rl-games"
PY_SCRIPT_PATH="/home/knum/projekty/rvae_reproduction/experiments/run.py"
# --- USER SPECIFIC CONSTANTS END ---

# Set script name
SCRIPT_NAME=$(basename "$0")
CURRENT_DATE=$(date +%Y-%m-%d)

# Print communicates
echo "Script name ${SCRIPT_NAME}"
echo "Current date ${CURRENT_DATE}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"
echo "PATH: $PATH"

# Print job information
echo "Current node: ${SLURM_NODELIST}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs allocated: ${SLURM_GPUS_PER_NODE}"

# Compute  info
echo "Number of CPU cores available: $(nproc)"
echo "Listing available GPUs with nvidia-smi:"
nvidia-smi
echo "Compact GPU list:"
nvidia-smi -L
echo "CUDA version:"
nvcc -V
# Print available GPU memory
echo "Available GPU memory:"
nvidia-smi --query-gpu=memory.total,memory.free --format=csv

# Activate the python virtual environment
source $ENV_PATH/bin/activate

# Add project directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/knum/projekty/rvae_reproduction"

python3 --version

# Run training script
python3 -u $PY_SCRIPT_PATH \
    --model "RVAE" \
    --dataset "fmnist" \
    --enc_layers 300 300 \
    --dec_layers 300 300 \
    --latent_dim 2 \
    --num_centers 350 \
    --num_components 5 \
    --device cuda \
    --sigma_learning_rate 0.01 \
    --warmup_learning_rate 0.01 \
    --save_dir "../saved_models/d2" 

echo "SCRIPT RUN FINISHED"