#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=2:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

cd ~/project2/depablo/erschultz
rm -r dataset_01_11_21 &
rm -r dataset_01_12_21 &

wait
