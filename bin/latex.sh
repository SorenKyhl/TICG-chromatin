#! /bin/bash
#SBATCH --job-name=latex
#SBATCH --output=logFiles/latex.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4000
#SBATCH --qos=depablo-debug

local='false'
if [ $local = 'true' ]
then
  dataDir='/home/erschultz/sequences_to_contact_maps'
  source activate python3.9_pytorch1.9
else
  dataDir='/project2/depablo/erschultz'
  source activate python3.9_pytorch1.9_cuda10.2
fi

dataset='dataset_04_27_22'
samples='1-2-3-4-5-6-7-8'
dataFolder="${dataDir}/${dataset}"
# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples
#

for sample in 1
 # 2 3 4 5 6 7 8
do
  python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample
done
