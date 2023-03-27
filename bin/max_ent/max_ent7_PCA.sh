#! /bin/bash
#SBATCH --job-name=maxent7
#SBATCH --output=logFiles/maxent7.out
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
  numIterations=12
  finalSimProductionSweeps=1000000
  equilibSweeps=100000
  productionSweeps=1000000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=6001
dataset='dataset_02_16_23'
useL='true'
useS='true'
useD='true'
m=512
chiMethod='zeros'
mode='both'

bondtype='gaussian'
bondLength=28 # TODO make sure this is correct !!!

diagChiMethod="zeros"
dense='true'
diagBins=80
nSmallBins=64
smallBinSize=1
diagCutoff=512

method='PCA-normalize'
jobs=0
waitCount=0
for k in 4
do
  for sample in 1 2 3 4 5 324 981 1936 2834 3464
  do
    gridSize="${dir}/${dataset}/samples/sample${sample}/config.json"
    echo "$sample m=$m k=$k"
    max_ent
    jobs=$(( $jobs + 1 ))
    if [ $jobs -gt 10 ]
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
