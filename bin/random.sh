#! /bin/bash
#SBATCH --job-name=TICG
#SBATCH --output=TICG.out
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
dataFolder="/project2/depablo/erschultz/dataset_10_08_21"
samplesPerTask=40
startSample=1
relabel='AB-D'

cd ~/TICG-chromatin/src
make
mv TICG-engine ..

echo $dataFolder
STARTTIME=$(date +%s)
for i in $(seq 1 40)
do
  start=$(( $(( $(( $i-1 ))*50 ))+$startSample ))
  stop=$(( $start+$samplesPerTask-1 ))
  echo $start $stop
  ~/TICG-chromatin/bin/random_inner.sh $i $k $chi $m $start $stop $dataFolder $relabel > ~/TICG-chromatin/logFiles/TICG${i}.log &
done

wait
ENDTIME=$(date +%s)
echo "total time: $(($ENDTIME-$STARTTIME)) seconds"
