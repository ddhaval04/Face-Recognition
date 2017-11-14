#!/bin/sh

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem-per-cpu=64000M
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:2

python script.py

