#! /bin/bash
#SBATCH --job-name=maxent3
#SBATCH --output=logFiles/maxent3.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

local='true'
source ~/TICG-chromatin/bin/max_ent/max_ent_fns.sh

if [ $local = 'true' ]
then
  dir="/home/erschultz/sequences_to_contact_maps"
  scratchDir='/home/erschultz/scratch'
  numIterations=15
  finalSimProductionSweeps=1000000
  equilibSweeps=100000
  productionSweeps=1000000
  source activate python3.9_pytorch1.9
fi

STARTTIME=$(date +%s)
i=2000
dataset='single_cell_nagano_imputed'
useE='false'
method='none'
diagChiMethod='zero'
chiDiagSlope=1
mode='diag'
dense='true'
bondtype='gaussian'
bondLength=28
m=1024
replicate=1
maxDiagChi=10

diagBins=32
nSmallBins=16
smallBinSize=4
diagStart=0
diagCutoff=1024
MLPModelID=13

for sample in 443 244 390 65 84
do
	for k in 0
	do
	  echo $sample $m
 	  max_ent
	done
done

wait

ENDTIME=$(date +%s)
echo "total time:$(( $(( $ENDTIME - $STARTTIME )) / 60 )) minutes"
