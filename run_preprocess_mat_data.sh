#!/bin/bash
#SBATCH --account=staff
#SBATCH --job-name=bucket
#SBATCH --cpus-per-task=3         # CPU cores/threads
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-node=1
#SBATCH --output=preprocess.log
#SBATCH --partition=lvlWork
bash /home/gfang/work/FastSpeech2Epoch2/preprocess_mat_data.sh