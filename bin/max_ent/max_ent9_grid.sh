#! /bin/bash
#SBATCH --job-name=maxent9
#SBATCH --output=logFiles/maxent9.out
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
  numIterations=8
  finalSimProductionSweeps=50000
  equilibSweeps=10000
  productionSweeps=50000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=8002
dataset='dataset_01_26_23'
useL='false'
useS='false'
useE='false'
useD='false'
m=512
chiMethod='none'
mode='grid_size'

bondtype='gaussian'
phiChromatin=0.06
bondLength=16.5
gridSize=17.5

diagChiMethod="none"
dense='false'
method='none'

trust_region=5
gamma=0.2
jobs=0
waitCount=0
k=0
for sample in {201..292}
do
  echo $sample $m
  max_ent
  jobs=$(( $jobs + 1 ))
  if [ $jobs -gt 17 ]
  then
    echo 'Waiting'
    waitCount=$(( $waitCount + 1 ))
    wait
    jobs=0
  fi
done

echo $waitCount
wait


ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
