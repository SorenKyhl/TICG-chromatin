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
  dir="/home/erschultz/sequences_to_contact_maps"
  scratchDir='/home/erschultz/scratch'
  numIterations=1
  finalSimProductionSweeps=500000
  productionSweeps=1000
  equilibSweeps=100
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=3003
dataset='dataset_07_20_22'
useS='false'
useE='true'
useD='false'
m=1024
GNNModelID=243
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

k=0
method='GNN'
for sample in 10
 # 1 1128 1131 1794 552 1938
do
  echo $sample $m
  echo $CONDA_DEFAULT_ENV
  max_ent
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
