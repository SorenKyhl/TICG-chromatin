#! /bin/bash
#SBATCH --job-name=max_ent
#SBATCH --output=logFiles/max_ent.out
#SBATCH --time=24:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo
#SBATCH --ntasks=48
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=END
#SBATCH --mail-user=erschultz@uchicago.edu


cd ~/TICG-chromatin
source activate python3.9_pytorch1.9_cuda10.2

python3 ~/TICG-chromatin/scripts/max_ent.py
