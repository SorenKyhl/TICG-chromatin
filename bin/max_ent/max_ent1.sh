#! /bin/bash
#SBATCH --job-name=maxent1
#SBATCH --output=logFiles/maxent1.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000

local='true'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/eric"
  scratchDir='/home/eric/scratch'
  numIterations=80
  # finalSimProductionSweeps=1000
  # equilibSweeps=1000
  # productionSweeps=10000
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  dir='/project2/depablo/erschultz'
  scratchDir='/scratch/midway2/erschultz'
  source activate python3.8_pytorch1.8.1_cuda10.2_2
fi

STARTTIME=$(date +%s)
i=1
dataset='dataset_test'

trust_region=10
# useE='true'
for method in 'RPCA-normalize'
do
  for sample in 85 86 87 88 89
  do
    for k in 2
    do
      max_ent
    done
  done
done


wait

# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
