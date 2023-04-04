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
i=6001

# nonbonded plaid
useL='true'
useS='true'
useD='true'
chiMethod='zeros'
method='PCA-normalize'
# nonbonded diag
diagChiMethod="zeros"
dense='true'
diagBins=96
nSmallBins=64
smallBinSize=1
diagCutoff=512
# bonded
bondtype='gaussian'
bondLength=16.5 # TODO make sure this is correct !!!
# newton's method
mode='both'
# bash
STARTTIME=$(date +%s)
jobs=0
waitCount=0

local='false'
if [ $local = 'true' ]
then
  dir="/home/erschultz"
  scratchDir='/home/erschultz/scratch'
  source activate python3.9_pytorch1.9
fi

# MC
numIterations=12
finalSimProductionSweeps=500000
equilibSweeps=100000
productionSweeps=500000

dataset='dataset_03_21_23'
m=512

for k in 8
do
  for sample in {1392..1450}
  do
    gridSize="${dir}/${dataset}/samples/sample${sample}/none/k0/replicate1/grid_size.txt"
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
done

echo $waitCount
wait


ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
