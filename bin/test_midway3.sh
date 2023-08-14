#! /bin/bash
#SBATCH --job-name=test
#SBATCH --output=test_midway3_0372.out
#SBATCH --account=pi-depablo
#SBATCH --gres=gpu:1
#SBATCH --partition=depablo-gpu
#SBATCH --mail-type=END
#SBATCH --mail-user=erschultz@uchicago.edu
#SBATCH --nodelist=midway3-0372

cd ~/TICG-chromatin

nvidia-smi

source activate python3.9_pytorch1.9
python3 test_midway3.py
conda deactivate

source activate pytorch_H100
python3 test_midway3.py
