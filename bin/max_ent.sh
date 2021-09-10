#! /bin/bash
#SBATCH --job-name=TICG_maxent
#SBATCH --output=TICG_maxent.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=7
#SBATCH --mem-per-cpu=2000

m=1024
k=4
sample=1201
dataFolder='/project2/depablo/erschultz/dataset_08_26_21'
productionSweeps=50000
equilibSweeps=10000
goalSpecified=1
numIterations=5 # iteration 1 + numIterations is production run to get contact map
overwrite=1
scratchDir='/scratch/midway2/erschultz/TICG_maxent'

source activate python3.8_pytorch1.8.1_cuda10.2
module load jq

STARTTIME=$(date +%s)
i=0
#'GNN' 'ground_truth' 'random' 'k_means' 'PCA' 'PCA_split' 'nmf'
for method in 'k_means' 'PCA' 'PCA_split'
do
  ~/TICG-chromatin/bin/max_ent_inner.sh $m $k $sample $dataFolder $productionSweeps $equilibSweeps $goalSpecified $numIterations $overwrite "${scratchDir}${i}" &
  i=$(($i + 1))
done

wait
ENDTIME=$(date +%s)
echo "total time: $(($ENDTIME-$STARTTIME)) seconds"
