#!/bin/bash
#SBATCH --job-name onnx_retraining_with_C++
#SBATCH -N 1
#SBATCH -p medium96s
#SBATCH -n 1
#SBATCH --time=24:00:00

./build/trainc++/trainc++ 60 ondevice
