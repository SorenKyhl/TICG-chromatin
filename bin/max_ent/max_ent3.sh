#! /bin/bash
#SBATCH --job-name=maxent3
#SBATCH --output=logFiles/maxent3.out
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
  numIterations=1
  finalSimProductionSweeps=5000
  productionSweeps=1000
  equilibSweeps=100
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=2001
dataset='dataset_11_14_22'
useS='false'
useE='true'
useD='false'
m=1024
GNNModelID=254
chiMethod='none'
mode='none'

bondtype='gaussian'
bondLength=28

diagChiMethod='none'
dense='false'
diagBins=1
nSmallBins=16
smallBinSize=4
diagCutoff=1024

k=0
method='GNN-kr'
for sample in 1
 # 2 3 4 5
do
  echo $sample $m
  max_ent
done
wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
