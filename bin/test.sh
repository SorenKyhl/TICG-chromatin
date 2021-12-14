#! /bin/bash
#SBATCH --job-name=test
#SBATCH --output=logFiles/contact_map.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000


source activate python3.8_pytorch1.8.1_cuda10.2

python3 ~/TICG-chromatin/scripts/test.py
