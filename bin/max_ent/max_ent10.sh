#! /bin/bash
#SBATCH --job-name=maxent10
#SBATCH --output=logFiles/maxent10.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

local='false'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/erschultz/sequences_to_contact_maps"
  scratchDir='/home/erschultz/scratch'
  numIterations=1
  finalSimProductionSweeps=4000
  productionSweeps=4000
  equilibSweeps=2000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=9000


wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
