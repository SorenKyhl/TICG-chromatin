#! /bin/bash
#SBATCH --job-name=test

#SBATCH --time=12:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=END
#SBATCH --mail-user=erschultz@uchicago.edu

cd ~/TICG-chromatin
source activate python3.9_pytorch1.9_cuda10.2
source activate python3.9_pytorch1.9

python3 ~/TICG-chromatin/scripts/test.py

# cd "/home/erschultz/dataset_04_28_23/setup"
# odir="/home/erschultz/dataset_04_28_23_repeat/setup"
#
# for i in 1 2 3 4 5 324 981 1936 2834 3464
# do
#   cp "chi_${i}.npy" "$odir/"
#   cp "diag_chis_${i}.npy" "$odir/"
#   cp "x_${i}.npy" "$odir/"
#   cp "sample_${i}.txt" "$odir/"
# done
