#!/bin/bash
#SBATCH --job-name=diffusion_train_condition_1    # Job name
#SBATCH --output=diffusion_%j.log     # Output file name (%j expands to jobID)
#SBATCH --error=diffusion_%j.err      # Error file name (%j expands to jobID)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH -p nvidia
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=32G                     # Memory pool for all cores
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=tt2273@nyu.edu    

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Load required modules
module load cuda/11.7
module load python/3.9
module load pytorch/2.0.0

# Activate conda environment
source activate ddpm_env

# Set environment variables
export PYTHONUNBUFFERED=1

# Run the training script directly from its location
# Note: We're assuming the script is in the current directory
python sample_power_spectra.py

# Print end time
echo "End time: $(date)"

# Print completion message
echo "Job completed successfully"