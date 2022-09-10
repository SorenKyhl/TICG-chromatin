#! /bin/bash
#SBATCH --job-name=maxent2
#SBATCH --output=logFiles/maxent2.out
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
  numIterations=15
  finalSimProductionSweeps=1000000
  productionSweeps=800000
  equilibSweeps=50000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=1010
dataset='dataset_07_20_22'
useE='false'
diagChiMethod='zero'
chiMethod='zero'
chiDiagSlope=1
mode='both'
dense='true'
bondtype='gaussian'
m=-1
replicate=1
maxDiagChi=10
parallel='false'
numThreads=1
trust_region=100

diagBins=32
nSmallBins=16
smallBinSize=4
diagStart=0
diagCutoff='none'


method='PCA-normalize'
for sample in 11 12 13 14
do
  for k in 5
  do
    max_ent
  done
done


wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
