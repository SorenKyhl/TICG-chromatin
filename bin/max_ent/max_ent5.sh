#! /bin/bash
#SBATCH --job-name=maxent5
#SBATCH --output=logFiles/maxent5.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000

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
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  dir='/project2/depablo/erschultz'
  scratchDir='/scratch/midway2/erschultz'
  source activate python3.8_pytorch1.8.1_cuda10.2_2
fi

STARTTIME=$(date +%s)
i=4000
dataset='dataset_01_15_22'
sample=40

useE='true'

for project in 'true' 'false'
do
  loadChi='true'
  method='PCA'
  k=4
  max_ent

  loadChi='false'
  method='GNN'
  modelID=70
  max_ent
done

wait

# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
