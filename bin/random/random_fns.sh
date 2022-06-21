#! /bin/bash

module unload gcc # not sure if this is necessary
module load gcc/10.2.0


param_setup() {
	# other params
	method='random'
	exclusive='false'
	overwrite=1
	dumpFrequency=50000
	e='none'
	s='none'
	chiConstant=0
	chiMultiplier=1
	chiDiagConstant=0
	sConstant=0
	pSwitch='none' # using lmbda instead
	npySeed='none'
	TICGSeed='none'
	if [ $chi = 'nonlinear' ]
	then
		useE='true'
		useS='false'
	elif [ $chi = 'polynomial' ]
	then
		useE='true'
		useS='false'
	else
		useE='false'
		useS='false'
	fi
}

move() {
	# move utils to scratch
	mkdir -p $scratchDirI
	cd ~/TICG-chromatin/utils
	init_config="input${m}.xyz"
	if [ -f $init_config ]
	then
		cp $init_config "${scratchDirI}/${init_config}"
	else
		init_config='none'
	fi
	cp default_config.json "${scratchDirI}/default_config.json"

	cd $scratchDirI
}

check_dir() {
	echo $i
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
}

random_inner() {
	# generate sequences
	python3 ~/TICG-chromatin/scripts/get_seq.py --method $method --exclusive $exclusive --m $m --p_switch $pSwitch --lmbda $lmbda --k $k --save_npy --seed $npySeed > seq.log

	# set up config.json
	python3 ~/TICG-chromatin/scripts/get_config.py --save_chi --chi=$chi --chi_seed $chiSeed --m $m --min_chi $minChi --max_chi $maxChi --fill_diag $fillDiag --ensure_distinguishable --diag $diag --max_diag_chi $maxDiagChi --n_sweeps $nSweeps --dump_frequency $dumpFrequency --TICG_seed $TICGSeed --use_ematrix $useE --use_smatrix $useS --load_configuration_filename $init_config --relabel $relabel --e $e --s $s --chi_constant=$chiConstant --chi_multiplier=$chiMultiplier --chi_diag_constant=$chiDiagConstant --s_constant=$sConstant > config.log

	# run simulation
	~/TICG-chromatin/TICG-engine > log.log

	# calculate contact map
	python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --k $k --save_npy --random_mode > contact_map.log

	# move inputs and outputs to own folder
	mkdir -p $dir
	mv config.json data_out *.log *.npy *.png *.txt $dir
}

random() {
	param_setup
	move

	for i in $(seq $start $stop)
	do
		check_dir
		random_inner
	done

	# clean up
	rm -f default_config.json *.xyz
}
