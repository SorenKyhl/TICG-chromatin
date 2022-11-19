#! /bin/bash
#SBATCH --job-name=maxent5
#SBATCH --output=logFiles/maxent5.out
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
  equilibSweeps=5000
  productionSweeps=5000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=4008
dataset='dataset_11_14_22'
useS='true'
useE='false'
useD='true'
m=1024
chiMethod='zero'
mode='both'

bondtype='gaussian'
bondLength=28
gridSize=27.5
beadVol=400
phiChromatin=0.06

diagChiMethod='zero'
dense='true'
diagBins=32
nSmallBins=16
smallBinSize=4
diagCutoff=1024

method='PCA_split-binarizeMean'
for k in 8
do
  for sample in 1
  do
    echo $sample $m
    echo $CONDA_DEFAULT_ENV
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
