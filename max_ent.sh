#! /bin/bash
#SBATCH --job-name=TICG
#SBATCH --output=TICG.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

m=1024
k=2
sample=40
dataFolder='/project2/depablo/erschultz/dataset_04_18_21'
sampleFolder="$dataFolder/samples/sample$sample"
scratchDir='/scratch/midway2/erschultz/TICG'
gamma=0.00001
gammaDiag=0.00001
mode="plaid"
productionSweeps=50000
equilibSweeps=10000
goalSpecified=1
numIterations=50

source activate python3.8_pytorch1.8.1_cuda10.2
module load jq

cd ~/TICG-chromatin/maxent/resources
python3 ~/TICG-chromatin/scripts/get_config.py --k $k --m $m --min_chi 0 --max_chi 0 --save_chi_for_max_ent

# 'PCA' 'k_means' 'GNN' 'random'
for method in 'ground_truth'
do
	cd ~/TICG-chromatin/maxent/resources
	# generate sequences
	python3 ~/TICG-chromatin/scripts/get_seq.py --method $method --m $m --k $k --sample $sample
	# generate goals
	python3 ~/TICG-chromatin/maxent/bin/get_goal_experimental.py --m $m --k $k --contact_map "$sampleFolder/y.npy"

	# apply max ent with newton's method
	dir="${dataFolder}/samples/sample${sample}/${method}/k${k}"
	~/TICG-chromatin/maxent/bin/run.sh $dir $gamma $gammaDiag $mode $productionSweeps $equilibSweeps $goalSpecified $numIterations

	# run longer simulation to estimate contact map
	cd $scratchDir
	~/TICG-chromatin/TICG-engine > log.log

  # compare results
  python3 ~/TICG-chromatin/scripts/compare_contact.py --m $m --sample $sample --data_folder $dataFolder
done
