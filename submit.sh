#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --account=su114-gpu


module purge
#module load GCC/10.2.0 CUDA/11.3.1  OpenMPI/4.0.5 Python/3.8.6
#source ~/torch/bin/activate
module load CUDA/11.7.0 GCCcore/11.3.0 GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
source ~/torch_1_13_1/bin/activate
cd ~/repositories/hypergraph-benchmarks
srun python train_all.py -t train
