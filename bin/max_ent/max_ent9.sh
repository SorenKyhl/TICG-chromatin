#! /bin/bash
#SBATCH --job-name=maxent9
#SBATCH --output=logFiles/maxent9.out
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
  equilibSweeps=50000
  productionSweeps=500000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=8000
dataset='dataset_9_30_22'
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
diagBins=32
nSmallBins=16
smallBinSize=4
diagCutoff=1024

k=4
method='k_means'
trust_region=10
for k in 4 6 8
do
  for sample in 552
  do
    echo $sample $m
    echo $CONDA_DEFAULT_ENV
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
