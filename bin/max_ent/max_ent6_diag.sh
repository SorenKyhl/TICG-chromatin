#! /bin/bash
#SBATCH --job-name=maxent7
#SBATCH --output=logFiles/maxent7.out
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
  finalSimProductionSweeps=50000
  equilibSweeps=10000
  productionSweeps=50000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=6010
dataset='dataset_02_04_23'
useL='false'
useS='false'
useE='false'
useD='true'
m=512
chiMethod='none'
mode='diag'

bondtype='gaussian'
bondLength=16.5
phiChromatin=0.06

diagChiMethod="zeros"
dense='true'
diagBins=64
nSmallBins=32
smallBinSize=1
diagCutoff=512

method='none'
jobs=0
waitCount=0
k=1
for sample in 201
do
  gridSize="${dir}/${dataset}/samples/sample${sample}/none/k0/replicate1/grid_size.txt"
  echo $sample $m
  max_ent
  jobs=$(( $jobs + 1 ))
  if [ $jobs -gt 18 ]
  then
    echo 'Waiting'
    waitCount=$(( $waitCount + 1 ))
    wait
    jobs=0
  fi
done

echo $waitCount
wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
