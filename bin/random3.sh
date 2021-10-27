#! /bin/bash
#SBATCH --job-name=TICG
#SBATCH --output=logFiles/random3.out
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

echo $@

STARTTIME=$(date +%s)
for i in $(seq 1 $tasks)
do
  start=$(( $(( $(( $i - 1 ))*$samples / $tasks ))+$startSample ))
  stop=$(( $start + $samplesPerTask - 1 ))
  i=$(( i + $tasks * $scriptIndex ))
  echo $start $stop
  scratchDirI="${scratchDir}/TICG${i}"
  ~/TICG-chromatin/bin/random_inner.sh $scratchDirI $k $chi $m $start $stop $dataFolder $relabel $diag > ~/TICG-chromatin/logFiles/TICG${i}.log &
done

wait
ENDTIME=$( date +%s )
echo "total time: $(( $ENDTIME - $STARTTIME )) seconds"


# clean up scratch
for i in $(seq 1 $tasks)
do
  i=$(( i + $tasks * $scriptIndex ))
  rm -d "${scratchDir}/TICG${i}"
done
