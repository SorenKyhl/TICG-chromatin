#! /bin/bash
#SBATCH --job-name=latex
#SBATCH --output=logFiles/latex.out
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

dataset='dataset_09_30_22'
samples='1-10-100-1000-1001-1002-1003'
dataFolder="${dataDir}/${dataset}"
# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples
#
#
for sample in 1 10 100 1000 1001 1002 1003
do
  python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample
done
