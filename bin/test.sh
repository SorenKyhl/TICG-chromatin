#! /bin/bash
#SBATCH --job-name=test
#SBATCH --output=logFiles/test.out
#SBATCH --time=12:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=broadwl
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=END
#SBATCH --mail-user=erschultz@uchicago.edu


source activate python3.9_pytorch1.9_cuda10.2
source activate python3.9_pytorch1.9

python3 ~/TICG-chromatin/scripts/test.py
