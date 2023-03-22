#! /bin/bash
#SBATCH --job-name=maxent6
#SBATCH --output=logFiles/maxent6.out
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
i=6501
dataset='dataset_03_03_23'
useL='true'
useS='true'
useD='true'
m=512
chiMethod='zeros'
mode='both'

bondtype='gaussian'
bondLength=16.5 # TODO make sure this is correct !!!

diagChiMethod="zeros"
dense='true'
diagBins=96
nSmallBins=64
smallBinSize=1
diagCutoff=512

method='PCA-normalize'
jobs=0
waitCount=0
for k in 4 8
do
  for sample in 1 2 3 4 5 324 981 1936 2834 3464
  do
    gridSize="${dir}/${dataset}/samples/sample${sample}/config.json"
    echo "$sample m=$m k=$k"
    max_ent
    jobs=$(( $jobs + 1 ))
    if [ $jobs -gt 8 ]
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
