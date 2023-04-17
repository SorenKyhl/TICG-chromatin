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
i=8002

# nonbonded
useL='false'
useS='false'
useD='false'
chiMethod='none'
method='none'
k=10
# bonded
bondLength=488
gridSize=600
beadVol=130000
phiChromatin=0.006
# newton's method
trust_region=20
gamma=2
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

numIterations=7
finalSimProductionSweeps=50000
equilibSweeps=20000
productionSweeps=50000

dataset='Su2020'
m=512

for sample in {1002..1002}
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
