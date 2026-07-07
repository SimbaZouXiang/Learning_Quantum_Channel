#!/bin/bash
#SBATCH -p pci
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 00:05:00
source ~/.virtualenvs/QIP/bin/activate
python benchmark_threads.py
