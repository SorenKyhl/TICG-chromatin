#! /bin/bash
#SBATCH --job-name=TICG15
#SBATCH --output=logFiles/random15.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2000

chi=$1
k=$2
m=$3
dataFolder=$4
startSample=$5
relabel=$6
tasks=$7
samples=$8
samplesPerTask=$9
diag=${10}
scratchDir=${11}
scriptIndex=${12}
nSweeps=${13}
pSwitch=${14}
minChi=${15}
maxChi=${16}
fillDiag=${17}
chiSeed=${18}
maxDiagChi=${19}


echo $@

source ~/TICG-chromatin/bin/random/random_fns.sh

STARTTIME=$(date +%s)
for i in $(seq 1 $tasks)
do
  start=$(( $(( $(( $i - 1 ))*$samples / $tasks ))+$startSample ))
  stop=$(( $start + $samplesPerTask - 1 ))
  i=$(( i + $tasks * $scriptIndex ))
  echo $start $stop
  scratchDirI="${scratchDir}/TICG${i}"
  random
done

wait
ENDTIME=$( date +%s )
echo "total time: $(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"


# clean up scratch
for i in $(seq 1 $tasks)
do
  i=$(( i + $tasks * $scriptIndex ))
  rm -d "${scratchDir}/TICG${i}"
done
