#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
# set max wallclock time
#SBATCH --time=48:00:00
# set name of job
#SBATCH --job-name=jobzl432
# set number of GPUs
#SBATCH --gres=gpu:1 -p small
# set RAM size
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=zl432@cantab.ac.uk
module load pytorch/1.12.1
source activate torch
cd /jmain02/home/J2AD015/axf03/zxl79-axf03/repository/hypergraph-benchmarks

CUDA_VISIBLE_DEVICES=0 python train_all.py -t train
