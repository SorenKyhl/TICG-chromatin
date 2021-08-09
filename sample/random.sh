#! /bin/bash
#SBATCH --job-name=TICG
#SBATCH --output=sample/TICG.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

method='random'
m=1024
pSwitch=0.05
k=4
numSimulations=1


cd ~/TICG-chromatin/sample
source activate python3.8_pytorch1.8.1_cuda10.2
for i in {1..$numSimulations}
do
	# generate sequences
	python3 get_seq.py --method $method --m $m --p_switch $pSwitch --k $k

	# run simulation
	./TICG-engine > log.log

  # calculate contact map
  python3 contactmap.py $i

	# move output to own folder
  dir='/project2/depablo/erschultz/dataset_08/09_21/samples/sample'$i
	mkdir -p $dir
	mv data_out log.log seq1.txt seq2.txt $dir

done
