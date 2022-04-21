#! /bin/bash
#SBATCH --job-name=maxent5
#SBATCH --output=logFiles/maxent5.out
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
  # finalSimProductionSweeps=5000
  # equilibSweeps=1000
  # productionSweeps=1000
  source activate python3.9_pytorch1.11
fi

STARTTIME=$(date +%s)
i=4000
dataset='dataset_09_02_21'
useE='true'
modelID=46
for method in 'GNN'
do
  for sample in 40 1230 1718
  do
    max_ent
  done
done


wait

wait

# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
