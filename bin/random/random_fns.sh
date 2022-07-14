#! /bin/bash

module unload gcc # not sure if this is necessary
module load gcc/10.2.0


param_setup() {
	# other params
	seqMethod='random'
	exclusive='false'
	pSwitch='none' # using lmbda instead
	seqSeed='none'

	overwrite=1
	dumpFrequency=50000
	e='none'
	s='none'
	sConstant=0
	useE='false'
	useS='false'

	constantChi=0

	chi='none'
	chiConstant=0
	chiMultiplier=1

	chiDiagConstant=0
	chiDiagSlope=1
	diagBins=20

	TICGSeed='none'
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
	python3 ~/TICG-chromatin/scripts/get_params.py --method $seqMethod --exclusive $exclusive --m $m --p_switch $pSwitch --lmbda $lmbda --k $k --save_npy --seq_seed $seqSeed --chi=$chi --chi_method $chiMethod --chi_seed $chiSeed --min_chi $minChi --max_chi $maxChi --fill_diag $fillDiag --ensure_distinguishable --diag_chi_method $chiDiagMethod --diag_chi_slope $chiDiagSlope --max_diag_chi $maxDiagChi --diag_bins $diagBins  --chi_constant=$chiConstant --chi_multiplier=$chiMultiplier --diag_chi_constant=$chiDiagConstant > params.log

	# set up config.json
	python3 ~/TICG-chromatin/scripts/get_config.py --m $m --n_sweeps $nSweeps --dump_frequency $dumpFrequency --TICG_seed $TICGSeed --constant_chi $constantChi --use_ematrix $useE --use_smatrix $useS --load_configuration_filename $init_config --relabel $relabel --e $e --s $s  > config.log

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
