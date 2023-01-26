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
  finalSimProductionSweeps=500000
  equilibSweeps=100000
  productionSweeps=500000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=6010
dataset='dataset_11_14_22'
useS='false'
useE='true'
useD='true'
m=1024
chiMethod='zeros'
mode='plaid'

bondtype='gaussian'
bondLength=28
phiChromatin=0.06

diagChiMethod='/home/erschultz/dataset_11_14_22/samples/sample2201/none/k0/replicate1/chis_diag.txt'
dense='true'
diagBins=32
nSmallBins=16
smallBinSize=4
diagCutoff=1024
constantChi=10

method='k_means'
for k in 3
do
  for sample in 2201
  # 2201 2202 2203 2204 2205 2206 2207 2208
  do
    echo $sample $m
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
