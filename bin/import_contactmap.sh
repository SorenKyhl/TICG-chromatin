#! /bin/bash
#SBATCH --job-name=import
#SBATCH --output=import.out
#SBATCH --time=2:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000

# activate python
source activate python3.8_pytorch1.8.1_cuda10.2

python3 ~/TICG-chromatin/scripts/import_contactmap_straw.py
