#! /bin/bash
#SBATCH --job-name=maxent6
#SBATCH --output=logFiles/maxent6.out
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
  numIterations=80
  # finalSimProductionSweeps=2000
  # equilibSweeps=1000
  # productionSweeps=5000
  source activate python3.9_pytorch1.11
fi

STARTTIME=$(date +%s)
i=5000
dataset='dataset_01_17_22'
useE='true'
diagPseudobeadsOn='false'
method='GNN'
modelID=137
for sample in 1 2 3 4 5 6 7 8
do
  max_ent
done

wait

# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
