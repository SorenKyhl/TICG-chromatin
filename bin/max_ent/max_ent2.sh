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
  numIterations=30
  finalSimProductionSweeps=1000000
  productionSweeps=1000000
  equilibSweeps=100000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=1000
dataset='dataset_07_20_22'
useE='false'
method='PCA-normalize'
diagChiMethod='linear'
chiDiagSlope=1
mode='both'
dense='true'
bondtype='gaussian'
m=1024
replicate=1
maxDiagChi=10

diagBins=50
nSmallBins=30
smallBinSize=4
diagStart=4
diagCutoff=1024

k=3
for sample in 8
do
  echo $sample $m
  max_ent_resume 25
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
