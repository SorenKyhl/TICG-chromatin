#! /bin/bash
#SBATCH --job-name=maxent7
#SBATCH --output=logFiles/maxent7.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo
#SBATCH --account=pi-depablo
#SBATCH --ntasks=24
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh
i=6010

# nonbonded plaid
useL='true'
useS='true'
useD='true'
chiMethod='zeros'
method='PCA-normalize'
# nonbonded diag
diagChiMethod="zeros"
dense='true'
diagBins=96
nSmallBins=64
smallBinSize=1
diagCutoff=1024
# bonded
bondtype='gaussian'
bondLength=223.93945336907979
gridSize=227.7170984734238
beadVol=260000.0
# newton's method
mode='both'
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

# MC
numIterations=12
finalSimProductionSweeps=3500
equilibSweeps=1000
productionSweeps=2500
dataset='dataset_test'
m=1024

for k in 8
do
  for sample in 5000
  do
    # gridSize="${dir}/${dataset}/samples/sample${sample}/none/k0/replicate1/grid_size.txt"
    echo "$sample m=$m k=$k"
    max_ent_resume 10
    jobs=$(( $jobs + 1 ))
    if [ $jobs -gt 22 ]
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
