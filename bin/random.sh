#! /bin/bash
#SBATCH --job-name=TICG
#SBATCH --output=TICG.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=2000

chi="-1&2&-1&1.5\\2&-1&-1&-0.5\\-1&-1&-1&1.5\\1.5&-0.5&1.5&-1"
# chi="-1&2\\2&-1"
k=4
m=1024
today=$(date +'%m_%d_%y')
dataFolder="/project2/depablo/erschultz/dataset_test"
samplesPerTask=2
startSample=1

for i in $(seq 1 16)
do
  start=$(( $(( $(( $i-1 ))*$samplesPerTask ))+$startSample ))
  stop=$($start+$samplesPerTask)
  echo $start $stop
  # ~/TICG-chromatin/bin/random_inner.sh $i $k $chi $m $start $stop $dataFolder & >> TICG${i}.log
done

wait


# sbatch bin/random1.sh 1 50 $k $chi $dataFolder
# sbatch bin/random2.sh 801 850 $k $chi $dataFolder
# sbatch bin/random3.sh 851 900 $k $chi $dataFolder
# sbatch bin/random4.sh 901 950 $k $chi $dataFolder
# sbatch bin/random5.sh 951 1000 $k $chi $dataFolder
# sbatch bin/random6.sh 1001 1050 $k $chi $dataFolder
# sbatch bin/random7.sh 1051 1100 $k $chi $dataFolder
# sbatch bin/random8.sh 1101 1150 $k $chi $dataFolder
# sbatch bin/random9.sh 1150 1200 $k $chi $dataFolder
# sbatch bin/random10.sh 451 500 $k $chi $dataFolder
# sbatch bin/random11.sh 501 550 $k $chi $dataFolder
# sbatch bin/random12.sh 551 600 $k $chi $dataFolder
# sbatch bin/random13.sh 601 650 $k $chi $dataFolder
# sbatch bin/random14.sh 651 700 $k $chi $dataFolder
# sbatch bin/random15.sh 701 750 $k $chi $dataFolder
# sbatch bin/random16.sh 751 800 $k $chi $dataFolder
