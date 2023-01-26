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
  numIterations=10
  finalSimProductionSweeps=500000
  productionSweeps=500000
  equilibSweeps=100000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=1
dataset='dataset_11_21_22'
useS='false'
useE='true'
useD='true'
m=1024
chiMethod='zeros'
mode='both'

bondtype='gaussian'
bondLength=28

diagChiMethod='zeros'
dense='true'
diagBins=96
nSmallBins=64
smallBinSize=1
diagCutoff=1024

method='k_means'
for sample in 410
 # 653 1462 1801 2290
do
  for k in 6
  # 4 8 12
  do
      echo $sample $m
      max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
