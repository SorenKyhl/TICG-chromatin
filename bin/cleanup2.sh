#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --output=logFiles/cleanup2.out
#SBATCH --time=2:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

cd ~/scratch-midway2

rm -r dataset* &
rm -r TICG* &

wait
