#! /bin/bash
#SBATCH --job-name=latex2
#SBATCH --output=logFiles/latex2.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4000
#SBATCH --qos=depablo-debug

local='true'
if [ $local = 'true' ]
then
  dataDir='/home/erschultz'
  source activate python3.9_pytorch1.9
else
  dataDir='/project2/depablo/erschultz'
  source activate python3.9_pytorch1.9_cuda10.2
fi

dataset='dataset_11_21_22'
samples='410-653-1462-1801-2290'
dataFolder="${dataDir}/${dataset}"
python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples
#
#
for sample in 410 653 1462 1801 2290
do
  python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample &
done

wait
