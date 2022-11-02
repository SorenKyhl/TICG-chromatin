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
  finalSimProductionSweeps=1000000
  productionSweeps=1000000
  equilibSweeps=100
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=3000
dataset='dataset_09_30_22'
useD='false'
m=1024
GNNModelID=223
chiMethod='none'
mode='none'

bondtype='gaussian'
bondLength=20

diagChiMethod='none'
dense='false'
diagBins=1
nSmallBins=16
smallBinSize=4
diagCutoff=1024
bondLength=20

k=0
method='GNN'
for sample in 1 1128 1131 1794 552 1938
do
  echo $sample $m
  max_ent
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
