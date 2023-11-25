#!/bin/bash
#SBATCH --gres=gpu:a5000:1  # Replace <number_of_gpus> with the number of GPUs you need
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:00:00           # Set the maximum time your job needs to run (e.g., 1:00:00 for one hour)
#SBATCH --job-name=my_gpu_job
#SBATCH --output=processor_output.out     # Custom output file for standard output
#SBATCH --error=processor_output.err      # Custom output file for standard error
#SBATCH --exclude=<node_hostname>        # Replace <node_hostname> with the hostname of the node you want to exclude

# Load required CUDA module

module load cuda/11.7
source /cs/snapless/gabis/shaharspencer/myGPUenv/bin/activate
module load cuda/11.7


# Assuming you have a requirements.txt file listing all the dependencies
python masking_algorithm.py