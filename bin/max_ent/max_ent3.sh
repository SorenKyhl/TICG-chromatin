#! /bin/bash
#SBATCH --job-name=maxent3
#SBATCH --output=logFiles/maxent3.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000

local='true'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/eric/sequences_to_contact_maps"
  scratchDir='/home/eric/scratch'
  # numIterations=1
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
i=2000
dataset='dataset_11_14_21'


dataset='dataset_11_14_21'
method='k_means'
for j in 1 2 3 4
do
  for l in 40
   # 1230 1718 1751 1761
  do
    sample="${l}/ground_truth-rank${j}-E/knone/replicate1"
    for k in 4 6
    do
      max_ent
    done
  done
done

wait


python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
