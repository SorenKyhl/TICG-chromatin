#! /bin/bash
#SBATCH --job-name=TICG
#SBATCH --output=logFiles/random.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=50
#SBATCH --mem-per-cpu=2000

# chi="-1&2&-1&1.5\\2&-1&-1&-0.5\\-1&-1&-1&1.5\\1.5&-0.5&1.5&-1"
chi="-1&1&0&0\\1&-2&0&-1\\0&0&-1&2\\0&-1&2&-1"
# chi="-1&1\\1&0"
# chi='none'
k=4
m=1024
today=$(date +'%m_%d_%y')
dataFolder="/project2/depablo/erschultz/dataset_10_25_21"
startSample=1
relabel='none'
tasks=50
samples=1000
samplesPerTask=$(($samples / $tasks))
diag='false'
local=0

if [ $local -eq 1 ]
then
  dataFolder="/home/eric/dataset_test"
  tasks=1
fi

cd ~/TICG-chromatin/src
make
mv TICG-engine ..

echo $dataFolder
STARTTIME=$(date +%s)
for i in $(seq 1 $tasks)
do
  start=$(( $(( $(( $i-1 ))*$samples / $tasks ))+$startSample ))
  stop=$(( $start+$samplesPerTask-1 ))
  echo $start $stop
  ~/TICG-chromatin/bin/random_inner.sh $i $k $chi $m $start $stop $dataFolder $relabel $diag $local > ~/TICG-chromatin/logFiles/TICG${i}.log &
done

wait
ENDTIME=$(date +%s)
echo "total time: $(( $ENDTIME-$STARTTIME )) seconds"

if [ $local -eq 0 ]
then
  # clean up scratch
  for i in $(seq 1 50)
  do
    rm -d "/scratch/midway2/erschultz/TICG${i}"
  done
fi
