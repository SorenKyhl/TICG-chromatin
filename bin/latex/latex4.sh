#! /bin/bash
#SBATCH --job-name=latex4
#SBATCH --output=logFiles/latex4.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem=0
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

dataset='dataset_04_05_23'
samples='1001-1002-1003-1004-1005-1006-1007-1008-1009-1010'
# samples='289-290-291-292'
dataFolder="${dataDir}/${dataset}"
convergenceDefinition='all'
python3 ~/TICG-chromatin/scripts/makeLatexTable_new.py --data_folder $dataFolder --samples $samples --convergence_definition $convergenceDefinition --experimental --convergence_mask


# for sample in 2217
# do
#   python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample --convergence_definition $convergenceDefinition --experimental &
# done

wait
