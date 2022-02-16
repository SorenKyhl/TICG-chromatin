#! /bin/bash
k=4
m=1024
dataFolder="/home/eric/dataset_test"
scratchDir='/home/eric/scratch'
useE='false'
useS='false'
startSample=1
relabel='none'
diag='true'
nSweeps=1000000
pSwitch=0.04
maxDiagChi=0.1
overwrite=1
dumpFrequency=50000

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
	python3 ~/TICG-chromatin/scripts/get_seq.py --method 'random' --exclusive 'false' --m $m --p_switch $pSwitch --k $k --save_npy --seed 'none'  >> seq.log

	# set up config.json
	python3 ~/TICG-chromatin/scripts/get_config.py --save_chi --e $e --chi=$chi --m $m --k $k --ensure_distinguishable --diag $diag --max_diag_chi $maxDiagChi --relabel $relabel --n_sweeps $nSweeps --dump_frequency $dumpFrequency --use_ematrix $useE --use_smatrix $useS --load_configuration_filename $init_config --TICG_seed 47 > config.log

	# run simulation
	~/TICG-chromatin/TICG-engine >> log.log

	# calculate contact map
	python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --save_npy --random_mode

	# move inputs and outputs to own folder
	mkdir -p $dir
	mv config.json data_out *.log *.npy *.png *.txt $dir

	# clean up
	rm default_config.json *.xyz

	rm $scratchDiri
}

# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

dir='/home/eric/sequences_to_contact_maps/dataset_11_14_21/samples/sample40'
useE='true'
chi='none'

e="${dir}/e.npy"
for i in 60 61
do
	run &
done

e="${dir}/PCA_analysis/e_1.npy"
i=62
run &

e="${dir}/PCA_analysis/e_2.npy"
i=63
run &

e="${dir}/PCA_analysis/e_3.npy"
i=64
run &


wait
