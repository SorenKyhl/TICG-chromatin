#! /bin/bash
#SBATCH --job-name=maxent11
#SBATCH --output=logFiles/maxent11.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

local='true'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/erschultz"
  scratchDir='/home/erschultz/scratch'
  numIterations=100
  productionSweeps=500000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=10000
dataset='dataset_test'
sample=10
mode='plaid'
method='PCA'
for k in 1 2 4
do
  max_ent
done

wait


ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
