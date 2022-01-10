#! /bin/bash
chi="-1&1&0&0\\1&-2&0&-1\\0&0&-1&2\\0&-1&2&-1"
k=4
m=1024
dataFolder="/home/eric/dataset_test"
scratchDir='/home/eric/scratch'
startSample=1
relabel='none'
nodes=10
tasks=20
samples=2000
diag='false'
nSweeps=50000
pSwitch=0.05
maxDiagChi=0.1
overwrite=1
dumpFrequency=1000

source activate python3.8_pytorch1.8.1_cuda11.1

run()  {
	echo $i

	# move utils to scratch
	scratchDiri="${scratchDir}/${i}"
	mkdir -p $scratchDiri
	cd ~/TICG-chromatin/utils
	init_config="input${m}.xyz"
	if [ -f $init_config ]
	then
		cp input1024.xyz "${scratchDiri}/input1024.xyz"
	else
		init_config='none'
	fi
	cp default_config.json "${scratchDiri}/default_config.json"

	cd $scratchDiri

	dir="${dataFolder}/samples/sample${i}"
	# directory checks
	if [ -d $dir ]
	then
		if [ $overwrite -eq 1 ]
		then
			echo "output directory already exists - overwriting"
			rm -r $dir
		else
			# don't overrite previous results!
			echo "output directory already exists - aborting"
			exit 1
		fi
	fi

	# generate sequences
	python3 ~/TICG-chromatin/scripts/get_seq.py --method 'random' --exclusive 'true' --m $m --p_switch $pSwitch --k $k --save_npy --seed 12 >> log.log

	# set up config.json
	python3 ~/TICG-chromatin/scripts/get_config.py --save_chi --chi=$chi --m $m --k $k --ensure_distinguishable --diag $diag --max_diag_chi $maxDiagChi --n_sweeps $nSweeps --dump_frequency $dumpFrequency --use_ematrix $useE --use_smatrix $useS --load_configuration_filename $init_config > log.log

	# run simulation
	~/TICG-chromatin/TICG-engine >> log.log

	# calculate contact map
	python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --save_npy

	# move inputs and outputs to own folder
	mkdir -p $dir
	mv config.json data_out log.log *.npy *.png *.txt $dir

	# clean up
	rm default_config.json *.xyz
}

# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

i=80
useE=False
useS=False
run &

wait
