#! /bin/bash
#SBATCH --job-name=TICG
#SBATCH --output=sample/TICG.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

m=1024
k=2
sample=40


cd ~/TICG-chromatin/sample
source activate python3.8_pytorch1.8.1_cuda10.2
for method in 'ground_truth' 'PCA' 'k_means' 'GNN'
do
	# generate sequences
	python3 get_seq.py --method $method --m $m --k $k --sample $sample

	# run simulation
	./TICG-engine > log.log # with max ent

  # calculate contact map
  python3 contactmap.py $i

  # compare function
  python3 compare_contact.py

	# move output to own folder
  dir='/project2/depablo/erschultz/dataset_08/09_21/samples/sample'$sample+'/'$method
	mkdir -p $dir
	mv data_out log.log seq1.txt seq2.txt $dir
  cp config.json $dir

done
