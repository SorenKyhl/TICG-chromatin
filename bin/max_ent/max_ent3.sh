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
  dir="/home/erschultz/sequences_to_contact_maps"
  scratchDir='/home/erschultz/scratch'
  numIterations=10
  finalSimProductionSweeps=800000
  productionSweeps=800000
  equilibSweeps=50000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=2003
dataset='dataset_07_20_22'
useE='false'
diagChiMethod='zero'
chiMethod='none'
mode='diag'
dense='true'
bondtype='gaussian'
bondLength=20
replicate=1
parallel='false'
numThreads=1
trust_region=100

diagBins=32
nSmallBins=16
smallBinSize=4
diagStart=0
diagCutoff='none'

method='none'
m=1024
for sample in 10
# 14
do
  for k in 0
  do
    echo $sample $m
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
