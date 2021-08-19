#! /bin/bash
#SBATCH --job-name=TICG_maxent
#SBATCH --output=TICG_maxent.out
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
gamma=0.00001
gammaDiag=0.00001
mode="plaid"
productionSweeps=50000
equilibSweeps=10000
goalSpecified=1
numIterations=1 # iteration 1 + numIterations is production run to get contact map
overwrite=1

OverallStartTime=$(date +%s)
source activate python3.8_pytorch1.8.1_cuda10.2
module load jq

# plot ref contact map
cd $sampleFolder
python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --ifile "y.npy"

# get config
cd ~/TICG-chromatin/maxent/resources
python3 ~/TICG-chromatin/scripts/get_config.py --k $k --m $m --min_chi 0 --max_chi 0 --fill_diag=-1 --save_chi_for_max_ent

# 'ground_truth' 'random' 'k_means' 'GNN'
for method in  'PCA'
do
	printf "\n${method}\n"
	cd ~/TICG-chromatin/maxent/resources
	# generate sequences
	python3 ~/TICG-chromatin/scripts/get_seq.py --method $method --m $m --k $k --sample $sample --data_folder $dataFolder
	# generate goals
	python3 ~/TICG-chromatin/maxent/bin/get_goal_experimental.py --m $m --k $k --contact_map "${sampleFolder}/y.npy"

	# apply max ent with newton's method
	dir="${sampleFolder}/${method}/k${k}"
	~/TICG-chromatin/maxent/bin/run.sh $dir $gamma $gammaDiag $mode $productionSweeps $equilibSweeps $goalSpecified $numIterations $overwrite

	# compare results
	cd $dir
	prodIt=$(($numIterations+1))
  python3 ~/TICG-chromatin/scripts/compare_contact.py --m $m --ifile1 "$sampleFolder/y.npy" --ifile2 "${dir}/iteration${prodIt}/y.npy"
done

EndTime=$(date +%s)
echo "total time: $(($EndTime - $OverallStartTime)) seconds"
