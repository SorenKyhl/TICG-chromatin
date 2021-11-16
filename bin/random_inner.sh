#! /bin/bash
scratchDir=${1}
k=$2
chi=$3
m=$4
startSimulation=$5
numSimulations=$6
dataFolder=$7
relabel=$8
diag=$9
nSweeps=${10}
pSwitch=${11}

# below does nothing if chi is given
minChi=${12}
maxChi=${13}
fillDiag=${14}

# other params
method='random'
maxDiagChi=0.1
overwrite=1
dumpFrequency=50000
if [ $chi = 'parity' ]
then
	useEnergy='true'
elif [ $chi = 'nonlinear' ]
then
	useEnergy='true'
else
	useEnergy='false'
fi

echo $@

# move utils to scratch
mkdir -p $scratchDir
cd ~/TICG-chromatin/utils
cp input1024.xyz "${scratchDir}/input1024.xyz"
cp default_config.json "${scratchDir}/default_config.json"

cd $scratchDir


for i in $(seq $startSimulation $numSimulations)
do
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

	# generate sequences
	python3 ~/TICG-chromatin/scripts/get_seq.py --method $method --m $m --p_switch $pSwitch --k $k --save_npy --relabel $relabel >> log.log

  # set up config.json
	python3 ~/TICG-chromatin/scripts/get_config.py --save_chi --chi=$chi --m $m --k $k --min_chi $minChi --max_chi $maxChi --fill_diag $fillDiag --ensure_distinguishable --diag $diag --max_diag_chi $maxDiagChi --n_sweeps $nSweeps --dump_frequency $dumpFrequency --seed $i --use_energy $useEnergy > log.log

	# run simulation
	~/TICG-chromatin/TICG-engine >> log.log

  # calculate contact map
  python3 ~/TICG-chromatin/scripts/contact_map.py --m $m --save_npy

	# move inputs and outputs to own folder
	mkdir -p $dir
	mv config.json data_out log.log *.npy *.png *.txt $dir
done

# clean up
rm default_config.json input1024.xyz
