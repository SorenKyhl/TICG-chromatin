#! /bin/bash
#SBATCH --job-name=TICG
#SBATCH --output=sample/TICG.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

cd ~/TICG-chromatin
source activate python3.8_pytorch1.8.1_cuda10.2
for i in 1
do
	# generate sequences
	python3 get_seq.py > seq1.txt
	python3 get_seq.py > seq2.txt

	# run simulation
	./TICG-engine > log.log

	# move output to own folder
	mkdir sample$i
	mv data_out log.log seq1.txt seq2.txt sample$i

	# calculate contact map
	python3 contactmap.py $i
done
