#! /bin/bash
#SBATCH --job-name=maxent6
#SBATCH --output=logFiles/maxent6.out
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
  numIterations=15
  finalSimProductionSweeps=500000
  equilibSweeps=100000
  productionSweeps=500000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=5001
dataset='dataset_11_14_22/samples/sample2217/PCA-normalize-E/k8/replicate1'
useS='false'
useE='true'
useD='true'
m=1024
chiMethod='zeros'
mode='both'

bondtype='gaussian'
bondLength=28

diagChiMethod='zeros'
dense='true'
diagBins=96
nSmallBins=64
smallBinSize=1
diagCutoff=1024

k=8
method='PCA-normalize'
for sample in 2217_copy 2217_neg_chi 2217_other_pcs 2217_rand_chi 2217_rand_seq 2217_shuffle_chi 2217_shuffle_seq
do
  echo $sample $m
  max_ent
done
wait

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
