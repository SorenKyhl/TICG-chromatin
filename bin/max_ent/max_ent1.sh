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
  numIterations=2
  finalSimProductionSweeps=5000
  productionSweeps=5000
  equilibSweeps=2000
  source activate python3.9_pytorch1.9
fi

numIterations=2
finalSimProductionSweeps=5000
productionSweeps=5000
equilibSweeps=2000

STARTTIME=$(date +%s)
i=1
dataset='dataset_05_18_22'
useE='false'
method='PCA-normalize'
diagChiMethod='linear'
mode='both'
seqSeed=12
chiSeed=12
TICGSeed=12
m=512

for sample in 1 2 3
do
  for k in 6
   # 2 4
  do
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
