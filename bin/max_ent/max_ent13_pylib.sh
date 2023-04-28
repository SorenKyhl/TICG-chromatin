#! /bin/bash
#SBATCH --job-name=maxent13
#SBATCH --output=logFiles/maxent13.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

local='false'
if [ $local = 'true' ]
then
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
jobs=0
waitCount=0
for sample in {201..282}
do
  echo $sample
  python3 ~/TICG-chromatin/max_ent.py $sample &
  jobs=$(( $jobs + 1 ))
  if [ $jobs -gt 16 ]
  then
    echo 'Waiting'
    waitCount=$(( $waitCount + 1 ))
    wait
    jobs=0
  fi
done

echo $waitCount
wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
