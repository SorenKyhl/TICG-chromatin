#! /bin/bash
#SBATCH --job-name=maxent4
#SBATCH --output=logFiles/maxent4.out
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
  numIterations=25
  finalSimProductionSweeps=1000000
  equilibSweeps=100000
  productionSweeps=1000000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=3000
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

diagBins=32
nSmallBins=16
smallBinSize=4
diagStart=0
diagCutoff='none'
k=3

sample=1
m=512
max_ent

sample=2
m=1024
max_ent

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
