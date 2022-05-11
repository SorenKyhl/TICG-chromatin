#! /bin/bash
#SBATCH --job-name=maxent12
#SBATCH --output=logFiles/maxent12.out
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
  dir="/home/erschultz/sequences_to_contact_maps"
  scratchDir='/home/erschultz/scratch'
  numIterations=60
  # finalSimProductionSweeps=1000
  equilibSweeps=20000
  productionSweeps=200000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=11000
dataset='dataset_04_27_22'
useE='false'
mode='plaid'
for method in 'PCA-normalize'
do
  for sample in 1
   # 2 3 4
  do
    for k in 5 6 10
    # 2 4 6 8
    do
      max_ent
    done
  done
done


wait

# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
