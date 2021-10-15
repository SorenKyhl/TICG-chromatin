#! /bin/bash
#SBATCH --job-name=TICG
#SBATCH --output=logFiles/move.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000


sourceDataFolder="/project2/depablo/erschultz/dataset_10_10_21"
outputDataFolder="/project2/depablo/erschultz/dataset_10_08_21"
samplesPerTask=10
startSample=41

echo $dataFolder
STARTTIME=$(date +%s)
for i in $(seq 1 40)
do
  start=$(( $(( $(( $i-1 ))*50 ))+$startSample ))
  stop=$(( $start+$samplesPerTask-1 ))
  for j in $(seq $start $stop)
  do
    mv "${sourceDataFolder}/samples/sample${j}" "${outputDataFolder}/samples/sample${j}"
  done
done

ENDTIME=$(date +%s)
echo "total time: $(($ENDTIME-$STARTTIME)) seconds"
