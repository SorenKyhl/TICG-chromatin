#! /bin/bash
#SBATCH --job-name=latex
#SBATCH --output=logFiles/latex.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4000
#SBATCH --qos=depablo-debug

source activate python3.8_pytorch1.8.1_cuda11.1

samples="40-1230-1718"
sample=40
# dataSet='dataset_08_24_21'
# dataSet='dataset_08_26_21'
# dataSet='dataset_08_29_21'
# dataSet='dataset_10_27_21'
# dataSet='dataset_11_03_21'
# dataSet='dataset_11_14_21'
# dataDir='/project2/depablo/erschultz'
dataDir='/home/eric/sequences_to_contact_maps'


dataset=dataset_11_14_21
# sample='81-82'
dataFolder="${dataDir}/${dataset}"
# python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples
#


for j in 1 2 3 4
# 1230 1718
do
  sample="40/ground_truth-rank${j}-E/knone/replicate1"
  python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample
done
