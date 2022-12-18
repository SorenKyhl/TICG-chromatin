#! /bin/bash
#SBATCH --job-name=maxent11
#SBATCH --output=logFiles/maxent11.out
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
  numIterations=15
  finalSimProductionSweeps=500000
  productionSweeps=500000
  equilibSweeps=50000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=10000
dataset='dataset_07_20_22'
useS='false'
useE='true'
useD='true'
m=1024
chiMethod='zero'
mode='both'

bondtype='gaussian'
bondLength=20
phiChromatin=0.06

diagChiMethod='zero'
dense='true'
diagBins=96
nSmallBins=64
smallBinSize=1
diagCutoff=1024

method='PCA-normalize'
for k in 8
# 12
do
  for sample in 2210
  do
    echo $sample $m
    max_ent
  done
done

wait


ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
