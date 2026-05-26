#!/bin/bash
#SBATCH --job-name=pytorch_v100_job      # Name of the job
#SBATCH --partition=gpufast              # Target the gpufast partition
#SBATCH --nodes=1                        # Request 1 node
#SBATCH --ntasks=1                       # Run a single task
#SBATCH --cpus-per-task=4                # CPU cores per task
#SBATCH --gres=gpu:v100:1                # Request 1 V100 GPU
#SBATCH --mem=8G                         # Memory limit
#SBATCH --time=01:00:00                  # Time limit (HH:MM:SS)
#SBATCH --output=logs_%j.out             # Standard output log (%j = job ID)
#SBATCH --error=logs_%j.err              # Standard error log

# 1. Clean the environment
# 2. Load the base toolchain
module purge
module load fosscuda/2020b

ml Anaconda3

unset PYTHONPATH
unset PYTHONHOME

# 3. Point DIRECTLY to your personal Miniconda install
# This bypasses the cluster's global 'conda' command
export PATH="/home/jaramand/miniconda3/bin:$PATH"

# 3. Use the direct path to the conda setup script
# (Change 'anaconda3' to 'miniconda3' if that's what you installed)
source /home/jaramand/miniconda3/etc/profile.d/conda.sh

# 4. Activate by name (or by full path if name fails)
conda activate /home/jaramand/miniconda3/envs/bidirectional_313

# 5. Double check activation in your logs
echo "Using Python from: $(which python)"
echo "Active environment: $CONDA_DEFAULT_ENV"

# 6. Run your code
~/miniconda3/envs/bidirectional_313/bin/python main.py 