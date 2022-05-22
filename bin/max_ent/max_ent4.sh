#! /bin/bash
#SBATCH --job-name=maxent4
#SBATCH --output=logFiles/maxent4.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

local='false'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/erschultz"
  scratchDir='/home/erschultz/scratch'
  numIterations=1
  finalSimProductionSweeps=5000
  equilibSweeps=1000
  productionSweeps=5000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=3000
dataset='dataset_09_21_21'
useE='true'
mode='both'
for method in 'GNN'
do
  for sample in 1 8 14 20
   # 2 3 4
  do
    for modelID in 149 150
    do
      max_ent
    done
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
