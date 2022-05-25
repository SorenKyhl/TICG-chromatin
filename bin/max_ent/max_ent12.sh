#! /bin/bash
#SBATCH --job-name=maxent12
#SBATCH --output=logFiles/maxent12.out
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
  dir="/home/erschultz"
  scratchDir='/home/erschultz/scratch'
  # numIterations=1
  # finalSimProductionSweeps=1000
  # equilibSweeps=200
  # productionSweeps=2000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=11000
dataset='dataset_05_18_22'
mode='both'
method='PCA-normalize'
for sample in 19
do
  for k in 2
  # 4 6 8
  do
    max_ent
  done
done


wait

# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
