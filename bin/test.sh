#! /bin/bash
#SBATCH --job-name=test
#SBATCH --output=logFiles/test.out
#SBATCH --time=24:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=END
#SBATCH --mail-user=erschultz@uchicago.edu


cd ~/TICG-chromatin
source activate python3.9_pytorch1.9_cuda10.2
source activate python3.9_pytorch1.9

# python3 ~/TICG-chromatin/scripts/test.py
python3 ~/TICG-chromatin/scripts/data_generation/test.py
