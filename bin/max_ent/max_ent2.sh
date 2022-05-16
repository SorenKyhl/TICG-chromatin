#! /bin/bash
#SBATCH --job-name=maxent2
#SBATCH --output=logFiles/maxent2.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

local='true'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/erschultz"
  scratchDir='/home/erschultz/scratch'
  # numIterations=1
  # finalSimProductionSweeps=1000
  # equilibSweeps=1000
  # productionSweeps=1000
  source activate python3.9_pytorch1.11
fi

# productionSweeps=200000
# equilibSweeps=20000
STARTTIME=$(date +%s)
i=1000
dataset='dataset_test'
# useE='true'
mode='both'
for method in 'PCA-normalize+constant'
do
  for sample in 12 15
   # 2 3 4
  do
    for k in 2 4
    do
      max_ent
    done
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
