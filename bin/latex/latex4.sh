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

dataset='dataset_01_26_23'
# samples='2217-2218-2219-2220-2221'
samples='283-284-285-286-287-288'
dataFolder="${dataDir}/${dataset}"
convergenceDefinition='strict'
python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --convergence_definition $convergenceDefinition --experimental &


# for sample in 2217
# do
#   python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample --convergence_definition $convergenceDefinition --experimental &
# done

wait
