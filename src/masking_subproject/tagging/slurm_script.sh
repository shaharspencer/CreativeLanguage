#!/bin/bash
#!/bin/bash
#SBATCH --gres=gpu:a5000:1  # Replace <number_of_gpus> with the number of GPUs you need
#SBATCH --time=40:00:00 # Set the maximum time your job needs to run (e.g., 1:00:00 for one hour)
#SBATCH --job-name=my_gpu_job
#SBATCH --output=processor_output.out     # Custom output file for standard output
#SBATCH --error=processor_output.out     # Custom output file for standard error
#SBATCH --mem=64GB          # Request 64 GB of memory

# Load required CUDA module

module load cuda/11.7
source /cs/snapless/gabis/shaharspencer/VENV_25_11/bin/activate
module load cuda/11.7

python -m spacy download en_core_web_lg


# Assuming you have a requirements.txt file listing all the dependencies
python masking_algorithm.py