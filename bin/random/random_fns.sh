#! /bin/bash

module unload gcc # not sure if this is necessary
module load gcc/10.2.0


param_setup() {
	# other params
	seqMethod='random'
	exclusive='false'
	pSwitch='none' # using lmbda instead
	lmbda='none'
	seqSeed='none'
	argsFile='none'
	scaleResolution=1

	overwrite=1
	dumpFrequency=50000
	dumpStatsFrequency=100
	trackContactMap='false'
	gridMoveOn='true'
	updateContactsDistance='false'
	eConstant=0
	sConstant=0
	useL='false'
	useS='false'
	useD='false'

	constantChi=0
	phiChromatin=0.06
	bondLength=16.5
	gridSize=28.7
	beadVol=520
	kAngle=0


	dense='false'
	denseCutoff='none'
	denseLoading='none'

	chi='none'
	chiMethod='none'
	k=0
	chiConstant=0
	chiMultiplier=1
	minChi=-1
	maxChi=1
	chiSeed='none'
	fillDiag='none'

	chiDiagMethod='none'
	diagBins=32
	chiDiagConstant=0
	chiDiagMidpoint=0
	chiDiagSlope=1
	chiDiagScale='none'
	mContinuous='none'
	smallBinSize=0
	bigBinSize=-1
	nSmallBins=0
	nBigBins=-1
	diagStart=0
	diagCutoff='none'
	minDiagChi=0
	maxDiagChi=10

	TICGSeed='none'
	bondType='gaussian'

	parallel='false'
	numThreads=2
}

move() {
	# move utils to scratch
	mkdir -p $scratchDirI
	cd ~/TICG-chromatin/defaults
	init_config="input${m}.xyz"
	if [ -f $init_config ]
	then
		cp $init_config "${scratchDirI}/${init_config}"
	else
		init_config='none'
	fi
	cp config.json "${scratchDirI}/default_config.json"

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
			echo "output directory already exists:"
			echo $dir
			echo "overwriting"
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
	python3 ~/TICG-chromatin/scripts/get_params.py --args_file $argsFile --method $seqMethod --exclusive $exclusive --m $m --p_switch $pSwitch --lmbda $lmbda --scale_resolution $scaleResolution --k $k --save_npy --seq_seed $seqSeed --chi=$chi --chi_method $chiMethod --chi_seed $chiSeed --min_chi $minChi --max_chi $maxChi --fill_diag $fillDiag --ensure_distinguishable --diag_chi_method $chiDiagMethod --diag_chi_slope $chiDiagSlope --diag_chi_scale $chiDiagScale --min_diag_chi $minDiagChi --max_diag_chi $maxDiagChi --diag_bins $diagBins --chi_constant=$chiConstant --chi_multiplier=$chiMultiplier --diag_chi_constant=$chiDiagConstant --diag_chi_midpoint=$chiDiagMidpoint --dense_diagonal_on $dense --dense_diagonal_cutoff $denseCutoff --dense_diagonal_loading $denseLoading --m_continuous $mContinuous --small_binsize $smallBinSize --big_binsize $bigBinSize --n_small_bins $nSmallBins --n_big_bins $nBigBins --diag_start $diagStart --diag_cutoff $diagCutoff > params.log

	# set up config.json
	python3 ~/TICG-chromatin/scripts/get_config.py --args_file $argsFile --phi_chromatin $phiChromatin --bond_length $bondLength --grid_size $gridSize --bead_vol $beadVol --k_angle $kAngle --m $m --n_sweeps $nSweeps --dump_stats_frequency $dumpStatsFrequency --dump_frequency $dumpFrequency --track_contactmap $trackContactMap --gridmove_on $gridMoveOn --update_contacts_distance $updateContactsDistance --TICG_seed $TICGSeed --constant_chi $constantChi --e_constant $eConstant --s_constant $sConstant --use_lmatrix $useL --use_smatrix $useS --use_dmatrix $useD --load_configuration_filename $init_config --bond_type $bondType --parallel $parallel --num_threads $numThreads --dense_diagonal_on $dense > config.log

	# run simulation
	~/TICG-chromatin/TICG-engine > log.log

	# calculate contact map
	python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --save_npy --random_mode --plot > contact_map.log

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
