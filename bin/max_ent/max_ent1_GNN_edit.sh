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
  dir="/home/erschultz"
  scratchDir='/home/erschultz/scratch'
  numIterations=1
  finalSimProductionSweeps=500000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=1
useL='false'
useS='true'
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
for sample in {201..210}
do
  dataset="dataset_02_04_23/samples/sample${sample}/PCA-normalize-E/k8/replicate1"
  gridSize="${dir}/dataset_02_04_23/samples/sample${sample}/none/k0/replicate1/grid_size.txt"
  # gridSize=24
  sample="${sample}_copy"
  for GNNModelID in 396
   # 382
  do
    echo $sample $m
    max_ent
  done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
