#! /bin/bash
#SBATCH --job-name=maxent10
#SBATCH --output=logFiles/maxent10.out
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
  numIterations=1
  finalSimProductionSweeps=2000
  productionSweeps=2000
  equilibSweeps=1000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=9000

dataset='dataset_09_21_21'
useE='false'
method='PCA-normalize'
diagChiMethod='linear'
maxDiagChi=20
m=1024
sample=1

mode='both'
bondType='DSS'
for k in 4 6
do
  max_ent
done

bondType='gaussian'
replicate=2
for k in 4 6
do
  max_ent
done

dense='true'
replicate=3
for k in 4 6
do
  max_ent
done

mode='all'
replicate=4
for k in 4 6
do
  max_ent
done

minDiagChi=0
replicate=5
for k in 4 6
do
  max_ent
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
