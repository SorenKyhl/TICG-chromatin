#! /bin/bash
#SBATCH --job-name=maxent8
#SBATCH --output=logFiles/maxent8.out
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
  equilibSweeps=100000
  productionSweeps=500000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=7001
dataset='dataset_01_26_23'
useL='true'
useS='false'
useE='true'
useD='true'
m=512
chiMethod='zeros'
mode='plaid'

bondtype='gaussian'
bondLength=28
phiChromatin=0.06

dense='true'
diagBins=96
nSmallBins=64
smallBinSize=1
diagCutoff=512


method='k_means'
jobs=0
waitCount=0
for k in 4
do
  for sample in {225..282}
  # 2201 2202 2203 2204 2205 2206 2207 2208 2209 2210 2211 2212 2213 2214 2215
  do
    diagChiMethod="/home/erschultz/dataset_01_26_23/samples/sample${sample}/none/k0/replicate1/chis_diag.txt"
    echo $sample $m
    max_ent
    jobs=$(( $jobs + 1 ))
    if [ $jobs -gt 11 ]
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
