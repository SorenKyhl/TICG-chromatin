#! /bin/bash
#SBATCH --job-name=maxent1
#SBATCH --output=logFiles/maxent1.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

local='true'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/erschultz"
  scratchDir='/home/erschultz/scratch'
  numIterations=3
  finalSimProductionSweeps=10000
  productionSweeps=10000
  equilibSweeps=2000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=1
dataset='dataset_test_diag512'
useE='false'
method='none'
diagChiMethod='mlp'
GNNModelID=159
MLPModelID=10
k=0
minDiagChi=0
for mode in 'none'
do
  for sample in 1
   # 2 3 4
  do
    max_ent
  done
done



wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
