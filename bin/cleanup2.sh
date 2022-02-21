#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --output=logFiles/cleanup2.out
#SBATCH --time=2:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

cd ~/scratch-midway2

rm -r TICG_maxent10* &
rm -r TICG_maxent11* &
rm -r TICG_maxent12* &
rm -r TICG_maxent13* &

wait
