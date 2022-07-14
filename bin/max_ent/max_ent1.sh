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
  numIterations=1
  finalSimProductionSweeps=100
  productionSweeps=2
  equilibSweeps=100
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=1
dataset='dataset_05_18_22'
useE='false'
method='PCA-normalize'
diagChiMethod='linear'
mode='diag'
dense='true'
bondtype='gaussian'
m=512
replicate=1
diagBins=10

for sample in 1
# 2 3
do
  for k in 7
  do
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
