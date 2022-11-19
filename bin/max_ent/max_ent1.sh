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
  numIterations=20
  finalSimProductionSweeps=1000000
  productionSweeps=1000000
  equilibSweeps=100000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=1
dataset='dataset_9_30_22'
bondType='gaussian'
m=1024
mode='both'
chiMethod='zero'

# diagChiMethod="${dir}/${dataset}/samples/sample1/none/k0/replicate1/chis_diag.txt"
diagChiMethod='zero'
dense='true'
diagBins=32
nSmallBins=16
smallBinSize=4
diagCutoff=1024
bondLength=20

k=6
method='PCA-normalize'
for sample in 1128 1131 1794 552 1938
do
  echo $sample $m
  max_ent
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
