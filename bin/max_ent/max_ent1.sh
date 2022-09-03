#! /bin/bash
#SBATCH --job-name=maxent1
#SBATCH --output=logFiles/maxent1.out
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
  finalSimProductionSweeps=1000000
  productionSweeps=1000000
  equilibSweeps=50000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=1
dataset='dataset_soren'
useE='false'
method='PCA-soren'
diagChiMethod='linear'
chiDiagSlope=1
mode='both'
dense='true'
bondtype='gaussian'
m=1024
replicate=1
maxDiagChi=10
parallel='false'
numThreads=1

diagBins=32
nSmallBins=16
smallBinSize=4
diagStart=0
diagCutoff=1024

k=5
sample=1
for method in PCA-normalize PCA-soren
 # 3 4 5
do
  echo $sample $m
  max_ent
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
