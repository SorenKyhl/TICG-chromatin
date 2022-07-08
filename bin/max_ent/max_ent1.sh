#! /bin/bash
#SBATCH --job-name=maxent1
#SBATCH --output=logFiles/maxent1.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

local='false'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/erschultz/sequences_to_contact_maps"
  scratchDir='/home/erschultz/scratch'
  numIterations=3
  finalSimProductionSweeps=5000
  productionSweeps=5000
  equilibSweeps=2000
  source activate python3.9_pytorch1.9
fi

numIterations=3
finalSimProductionSweeps=5000
productionSweeps=5000
equilibSweeps=2000

STARTTIME=$(date +%s)
i=1
dataset='dataset_05_18_22'
useE='false'
method='PCA-normalize'
diagChiMethod='linear'
k=2
mode='both'

m=512
for sample in 1 2 3
do
  max_ent
done

# m=1024
# for sample in 4 5 6
# do
#   max_ent
# done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
