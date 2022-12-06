#! /bin/bash
#SBATCH --job-name=test
#SBATCH --output=logFiles/test.out
#SBATCH --time=12:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000


source activate python3.9_pytorch1.9_cuda10.2

python3 ~/TICG-chromatin/scripts/test.py
