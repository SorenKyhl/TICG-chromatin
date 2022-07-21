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
  dir="/home/erschultz"
  scratchDir='/home/erschultz/scratch'
  numIterations=15
  finalSimProductionSweeps=1000000
  productionSweeps=300000
  equilibSweeps=50000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=1000
dataset='dataset_test'
useE='false'
method='none'
diagChiMethod='linear'
m=1024
mode='diag'
diagBins=32
k=0
maxDiagChi=15

dense='true'
sample=8
max_ent
#
dense='false'
sample=9
max_ent

m=512
sample=10
max_ent

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
