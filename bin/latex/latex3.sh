#! /bin/bash
#SBATCH --job-name=latex3
#SBATCH --output=logFiles/latex3.out
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

dataset='dataset_09_21_21'
dataFolder="${dataDir}/${dataset}"

for sample in 1 2 8 14 20
do
  python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample --experimental &
done

wait
