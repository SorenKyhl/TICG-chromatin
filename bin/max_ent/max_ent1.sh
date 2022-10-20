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
  numIterations=1
  finalSimProductionSweeps=10000
  productionSweeps=10000
  equilibSweeps=1000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=50
dataset='dataset_09_30_22'
useE='false'
useD='true'
GNNModelID='none'
bondType='gaussian'
m=1024
mode='none'
chiMethod='zero'
replicate=1

# diagChiMethod="${dir}/${dataset}/samples/sample1/none/k0/replicate1/chis_diag.txt"
diagChiMethod='zero'
useGroundTruthDiagChi='true'
maxDiagChi=10
dense='true'
diagBins=32
nSmallBins=16
smallBinSize=4
diagStart=0
diagCutoff=1024
bondLength=20

k=0
method=none
for sample in 1
# 10 100 1000 1001 1002 1003
do
  echo $sample $m
  max_ent
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
