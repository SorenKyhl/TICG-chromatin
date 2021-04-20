#!/bin/bash


for i in {1..250}
do
	# generate sequences
	python3 get_seq.py > seq1.txt
	python3 get_seq.py > seq2.txt

	# run simulation
	./TICG-engine > log.log

	# move output to own folder
	mkdir sample$i
	touch log.log
	mv data_out log.log sample$i

	# calculate contact map
	python3 contactmap.py $i
done


