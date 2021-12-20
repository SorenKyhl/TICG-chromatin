#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --output=logFiles/cleanup2.out
#SBATCH --time=2:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

cd ~/scratch-midway2

rm -r TICG_maxent1* &
rm -r TICG_maxent2* &
rm -r TICG_maxent3* &
rm -r TICG_maxent4* &
rm -r TICG_maxent5* &
rm -r TICG_maxent6* &
rm -r TICG_maxent7* &
rm -r TICG_maxent8* &
rm -r TICG_maxent9* &
