#! /bin/bash
#SBATCH --job-name=maxent7
#SBATCH --output=logFiles/maxent7.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo
#SBATCH --account=pi-depablo
#SBATCH --ntasks=24
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh
i=11002

# nonbonded plaid
useL='false'
useS='false'
useD='false'
chiMethod='none'
method='none'
k=10
# nonbonded diag
diagChiMethod="zeros"
dense='true'
diagBins=96
nSmallBins=64
smallBinSize=1
# bonded
bondLength=488
beadVol=130000
phiChromatin=0.006
# newton's method
mode='diag'
trust_region=300
# bash
STARTTIME=$(date +%s)
jobs=0
waitCount=0

local='true'
if [ $local = 'true' ]
then
  dir="/home/erschultz"
  scratchDir='/home/erschultz/scratch'
  source activate python3.9_pytorch1.9
fi

# MC
numIterations=15
finalSimProductionSweeps=200000
equilibSweeps=50000
productionSweeps=200000
dataset='Su2020'
m=512


for sample in 1002
do
  # gridSize=200
  gridSize="${dir}/${dataset}/samples/sample${sample}/none/k10/replicate1/grid_size.txt"
  echo "$sample m=$m k=$k"
  max_ent
  jobs=$(( $jobs + 1 ))
  if [ $jobs -gt 22 ]
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
