#!/bin/bash

#SBATCH --export=NONE
#SBATCH --partition=d0giesek
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=leezencounter
#SBATCH --output=/scratch/tmp/n_herr03/leezencounter/test.out
#SBATCH --error=/scratch/tmp/n_herr03/leezencounter/test.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n_herr03@uni-muenster.de
#SBATCH --mem=50GB

partition=d0giesek

dirname=$(date +"%Y-%m-%dT%H-%M-%S-${partition}")
module purge
ml palma/2022a  
ml GCCcore/11.3.0 
ml Python/3.10.4
ml Ninja/1.10.2
ml CUDA/11.7.0
ml OpenMPI/4.1.4
ml PyTorch/1.12.1-CUDA-11.7.0
pip install -r requirements.txt
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd ~/leezencounter/model-training/model_training/
pip install ppq==0.6.6
python __main__.py ../configs/yolo11n_sample_config.yaml
