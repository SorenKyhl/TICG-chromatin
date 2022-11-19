#! /bin/bash
#SBATCH --job-name=maxent10
#SBATCH --output=logFiles/maxent10.out
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
  finalSimProductionSweeps=500000
  productionSweeps=500000
  equilibSweeps=1000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=9000
dataset='dataset_11_14_22'
useS='false'
useE='false'
useD='false'
m=1024
chiMethod='none'
mode='none'

bondtype='gaussian'
bondLength=28
gridSize=26.6
phiChromatin=0.06

diagChiMethod='none'
dense='true'
diagBins=32
nSmallBins=16
smallBinSize=4
diagCutoff=1024

method='none'
for k in 0
do
  for sample in 1
  do
    echo $sample $m
    echo $CONDA_DEFAULT_ENV
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
