#! /bin/bash
#SBATCH --job-name=TICG
#SBATCH --output=TICG.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2000

chi="-1&2&-1&1.5\\2&-1&-1&-0.5\\-1&-1&-1&1.5\\1.5&-0.5&1.5&-1"
# chi="-1&2\\2&-1"
k=4
m=1024
today=$(date +'%m_%d_%y')
dataFolder="/project2/depablo/erschultz/dataset_08_26_21"
samplesPerTask=50
startSample=1

cd ~/TICG-chromatin/src
make
mv TICG-engine ..

STARTTIME=$(date +%s)
for i in $(seq 1 40)
do
  start=$(( $(( $(( $i-1 ))*$samplesPerTask ))+$startSample ))
  stop=$(( $start+$samplesPerTask-1 ))
  echo $start $stop
  ~/TICG-chromatin/bin/random_inner.sh $i $k $chi $m $start $stop $dataFolder > ~/TICG-chromatin/logFiles/TICG${i}.log &
done

wait
ENDTIME=$(date +%s)
echo "total time: $(($ENDTIME-$STARTTIME)) seconds"
