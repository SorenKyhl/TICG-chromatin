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
  numIterations=20
  finalSimProductionSweeps=1000000
  productionSweeps=1000000
  equilibSweeps=100000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=2001
dataset='dataset_07_20_22'
useE='false'
useD='true'
diagChiMethod='mlp'
MLPModelID='79'
chiMethod='zero'
mode='plaid'
dense='true'
bondtype='gaussian'
bondLength=20
replicate=1
parallel='false'
numThreads=1
trust_region=100

diagBins=1024
nSmallBins=16
smallBinSize=1
diagStart=0
diagCutoff='none'

method='PCA-normalize'
m=1024
for sample in 10
# 14
do
  for k in 3
  do
    echo $sample $m
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
