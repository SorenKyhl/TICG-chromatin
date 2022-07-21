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
  dir="/home/erschultz/sequences_to_contact_maps"
  scratchDir='/home/erschultz/scratch'
  numIterations=30
  finalSimProductionSweeps=1000000
  productionSweeps=500000
  equilibSweeps=50000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=1
dataset='dataset_07_20_22'
useE='false'
method='none'
diagChiMethod='linear'
mode='diag'
dense='true'
bondtype='gaussian'
m=-1
replicate=1

diagBins=32
denseCutoff=0.25
denseLoading=0.5

k=0
for sample in 1 2 3 4 5 6
do
  echo $sample $diagBins $denseCutoff
  max_ent
  denseCutoff=0.125
  diagBins=$(( $diagBins * 2 ))
  if [ $diagBins -gt 256 ]
  then
    denseCutoff=0.0625
    diagBins=256
  fi
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
