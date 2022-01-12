#! /bin/bash
#SBATCH --job-name=latex
#SBATCH --output=logFiles/latex.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --qos=depablo-debug

source activate python3.8_pytorch1.8.1_cuda10.2

# dataFolder='/project2/depablo/erschultz/dataset_09_21_21'
# samples=1
# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples

# dataFolder='/home/eric/dataset_test'
# sample='11'
# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample
# samples="11-13"
# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples

samples="40-1230-1718"
sample=40
# samples="51-52"
# dataSet='dataset_08_24_21'
# dataSet='dataset_08_26_21'
# dataSet='dataset_08_29_21'
# dataSet='dataset_10_27_21'
# dataSet='dataset_11_03_21'
# dataSet='dataset_11_14_21'
dataDir='/project2/depablo/erschultz'
# dataDir='/home/eric/sequences_to_contact_maps'

dataset=dataset_01_11_22
for sample in 40 41 42 43 44
do
  dataFolder="${dataDir}/${dataset}"
  python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample
done
