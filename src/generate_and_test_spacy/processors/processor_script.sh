#!/bin/bash
#SBATCH --gres=gpu:a5000:3  # Replace <number_of_gpus> with the number of GPUs you need
#SBATCH --cpus-per-task=11
#SBATCH --time=40:00:00           # Set the maximum time your job needs to run (e.g., 1:00:00 for one hour)
#SBATCH --job-name=my_gpu_job
#SBATCH --output=processor_output.out     # Custom output file for standard output
#SBATCH --error=processor_output.out     # Custom output file for standard error
#SBATCH --mem=6GB          # Request 64 GB of memory


# Load required CUDA module

export CUDA_LAUNCH_BLOCKING=1


module load cuda/11.7
source /cs/snapless/gabis/shaharspencer/creative_8123/bin/activate
module load cuda/11.7


# Assuming you have a requirements.txt file listing all the dependencies
python processor.py data_from_first_40000_posts_by_sentence.csv None False
