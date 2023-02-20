#! /bin/bash
#SBATCH --job-name=maxent2
#SBATCH --output=logFiles/maxent2.out
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
  finalSimProductionSweeps=500000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=1001
dataset='dataset_02_13_23'
useL='false'
useS='false'
useE='true'
useD='false'
m=512
chiMethod='none'
mode='none'

bondtype='gaussian'
bondLength=16.5

diagChiMethod='none'
dense='false'

k=0
method='GNN'
for sample in 324 981 1936 2834 3464
do
  gridSize="${dir}/${dataset}/samples/sample${sample}/config.json"
  for GNNModelID in 371 372
  # 271
   # 243 254 262 265 267 271 276
  do
    echo $sample $m
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
