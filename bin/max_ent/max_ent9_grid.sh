#! /bin/bash
#SBATCH --job-name=maxent9
#SBATCH --output=logFiles/maxent9.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo
#SBATCH --account=pi-depablo
#SBATCH --ntasks=24
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh
i=8001

# nonbonded
useL='false'
useS='false'
useD='false'
chiMethod='none'
method='none'
k=1
# bonded
bondLength=223.93945336907979
gridSize=227.7170984734238
beadVol=260000.0
# newton's method
trust_region=5
gamma=0.2
mode='grid_size'
# bash
STARTTIME=$(date +%s)
jobs=0
waitCount=0

local='true'
if [ $local = 'true' ]
then
  dir="/home/erschultz"
  scratchDir='/home/erschultz/scratch'
  source activate python3.9_pytorch1.9
fi

numIterations=5
finalSimProductionSweeps=50000
equilibSweeps=10000
productionSweeps=50000

dataset='dataset_test'
m=1024

for sample in 5000
do
  echo $sample $m
  max_ent
  jobs=$(( $jobs + 1 ))
  if [ $jobs -gt 22 ]
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
