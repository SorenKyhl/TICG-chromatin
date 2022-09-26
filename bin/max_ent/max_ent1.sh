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
  numIterations=0
  finalSimProductionSweeps=100000
  productionSweeps=80000
  equilibSweeps=5000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=1
dataset='dataset_04_27_22'
useE='true'
method='GNN'
GNNModelID=150
diagChiMethod="${dir}/${dataset}/samples/sample1/PCA-soren/k4/replicate1/chis_diag.txt"
diagChiMethod='none'
useGroundTruthDiagChi='true'
bondType='gaussian'
m=1024
replicate=1

dense='true'
diagBins=32
nSmallBins=16
smallBinSize=4
diagStart=0
diagCutoff=1024
bondLength=28

k=0
sample=1_10
for method in GNN
 # 3 4 5
do
  echo $sample $m
  max_ent
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
