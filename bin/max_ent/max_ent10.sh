#! /bin/bash
#SBATCH --job-name=maxent10
#SBATCH --output=logFiles/maxent10.out
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
  numIterations=5
  finalSimProductionSweeps=500000
  productionSweeps=500000
  equilibSweeps=100000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=9002
dataset='dataset_11_14_22'
useS='false'
useE='true'
useD='true'
m=1024
chiMethod='zero'
mode='both'

bondtype='gaussian'
bondLength=28
phiChromatin=0.06
trust_region=10000

diagChiMethod='zero'
dense='true'
diagBins=32
nSmallBins=16
smallBinSize=4
diagCutoff=1024

method='chromhmm-fold_change_control'
for k in 10
do
  for sample in 2201
  # 2203 2204 2205 2206 2207 2208 2209
  do
    echo $sample $m
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
