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

cd ~/TICG-chromatin/sample
source activate python3.8_pytorch1.8.1_cuda10.2

python3 get_config.py --k $k --m $m

for method in 'ground_truth' 'PCA' 'k_means' 'GNN' 'random'
do
	# generate sequences
	python3 get_seq.py --method $method --m $m --k $k --sample $sample

	# run simulation
	./TICG-engine > log.log # TODO max ent

  # calculate contact map
  python3 contactmap.py

  # compare results
  python3 compare_contact.py --m $m --sample $sample --data_folder $dataFolder

	# move output to own folder
  dir="${dataFolder}/samples/sample${sample}/${method}/k${k}"
	mkdir -p $dir
	mv data_out log.log x.npy y.npy y.png distance_pearson.png $dir
	for i in $(seq 0 $(($k-1)))
	do
		mv seq${i}.txt $dir
	done
done

mv config.json ${dataFolder}
