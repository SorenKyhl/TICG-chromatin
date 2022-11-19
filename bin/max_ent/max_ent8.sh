#! /bin/bash
#SBATCH --job-name=maxent8
#SBATCH --output=logFiles/maxent8.out
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
  numIterations=20
  finalSimProductionSweeps=500000
  equilibSweeps=50000
  productionSweeps=500000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=7000
dataset='dataset_11_14_22'
useS='false'
useE='true'
useD='true'
m=1024
chiMethod='zero'
mode='both'

bondtype='gaussian'
bondLength=28
phiChromatin=0.06

diagChiMethod='zero'
dense='true'
diagBins=32
nSmallBins=16
smallBinSize=4
diagCutoff=1024

k=4
method='PCA-binarize'
for k in 4 6
do
  for sample in 1
  # 5 6 9
  # 10 13 14 16 18
  do
    echo $sample $m
    echo $CONDA_DEFAULT_ENV
    max_ent
  done
done

wait


ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
