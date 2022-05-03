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
  dir="/home/erschultz"
  scratchDir='/home/erschultz/scratch'
  numIterations=1
  finalSimProductionSweeps=5000
  equilibSweeps=1000
  productionSweeps=1000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=4000
dataset='dataset_04_27_22'
useE='true'
method='ground_truth'
for sample in 1
 # 2 3 4 5 6 7 8
do
  max_ent
done


wait

wait

# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
