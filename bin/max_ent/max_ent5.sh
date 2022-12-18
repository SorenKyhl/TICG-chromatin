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
  finalSimProductionSweeps=500000
  equilibSweeps=500
  productionSweeps=5000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=4003
dataset='dataset_11_14_22/samples/sample2201/PCA-normalize-E/k8/replicate1'
useS='false'
useE='true'
useD='false'
m=1024
GNNModelID=289
chiMethod='none'
mode='none'

bondtype='gaussian'
bondLength=28

diagChiMethod='none'
dense='false'
diagBins=1
nSmallBins=16
smallBinSize=4
diagCutoff=1024

k=0
method='GNN'
for sample in 2201_shuffle_seq 2201_shuffle_chis 2201_rand_seq 2201_other_pcs 2201_neg_chi 2201_rand_chi
do
  echo $sample $m
  max_ent
done
wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
