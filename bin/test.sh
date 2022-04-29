#! /bin/bash
#SBATCH --job-name=test
#SBATCH --output=logFiles/test.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --qos=depablo-debug


source activate python3.9_pytorch1.9_cuda10.2

python3 ~/TICG-chromatin/scripts/test.py
