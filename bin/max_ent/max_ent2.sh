#! /bin/bash
#SBATCH --job-name=maxent2
#SBATCH --output=logFiles/maxent2.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

local='false'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/eric/sequences_to_contact_maps"
  scratchDir='/home/eric/scratch'
  # numIterations=1
  # finalSimProductionSweeps=1000
  # equilibSweeps=1000
  # productionSweeps=10000
  source activate python3.9_pytorch1.11
fi

STARTTIME=$(date +%s)
i=1000
dataset='dataset_01_17_22'
# useE='true'
diagPseudobeadsOn='false'
for method in 'PCA-normalize'
do
  for sample in 5 6 7 8
  do
    for k in 6 8
    do
      max_ent
    done
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
