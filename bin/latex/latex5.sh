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

dataset='dataset_02_04_23'
samples='201-202-203-204-205-206-207-208-209-210'
dataFolder="${dataDir}/${dataset}"
convergenceDefinition='normal'
python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --convergence_definition $convergenceDefinition --experimental &


# for sample in 201
# do
#   python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample --convergence_definition $convergenceDefinition --experimental &
# done

wait
