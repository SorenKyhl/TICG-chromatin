#! /bin/bash
#SBATCH --job-name=maxent3
#SBATCH --output=logFiles/maxent3.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

local='true'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/erschultz"
  scratchDir='/home/erschultz/scratch'
  numIterations=1
  finalSimProductionSweeps=1000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=2001
dataset='dataset_02_04_23'
useL='false'
useS='true'
useD='false'
m=512
chiMethod='none'
mode='none'
goalSpecified=0

bondtype='gaussian'
bondLength=140 # TODO make sure this is correct !!!
phiChromatin=0.03
beadVol=130000
TICGSeed=12
dumpFrequency=100
dumpStatsFrequency=1


k=0
method='GNN'
jobs=0
waitCount=0
for sample in 203
do
  for GNNModelID in 403
  do
    echo $sample $m
    max_ent
    jobs=$(( $jobs + 1 ))
    if [ $jobs -gt 19 ]
    then
      echo 'Waiting'
      waitCount=$(( $waitCount + 1 ))
      wait
      jobs=0
    fi
  done
done

echo $waitCount
wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
