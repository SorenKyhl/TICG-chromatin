#! /bin/bash
#SBATCH --job-name=maxent4
#SBATCH --output=logFiles/maxent4.out
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
  finalSimProductionSweeps=500000
  productionSweeps=500000
  equilibSweeps=100000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=3010
dataset='dataset_01_17_22'
useS='false'
useE='true'
useD='false'
m=1024
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
method='GNN'
for sample in 201 202 203 204 205
do
  for GNNModelID in 341
   # 243 254 262 265 267 271 276
  do
    echo $sample $m
    max_ent
  done
done
wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
