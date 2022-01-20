#! /bin/bash
#SBATCH --job-name=latex2
#SBATCH --output=logFiles/latex2.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4000
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

samples="40-1230-1718-1751-1761"
sample=40
ref='GNN'
# dataSet='dataset_08_24_21'
# dataSet='dataset_08_26_21'
# dataSet='dataset_08_29_21'
# dataSet='dataset_10_27_21'
# dataSet='dataset_11_03_21'
# dataSet='dataset_11_14_21'
dataDir='/project2/depablo/erschultz'
# dataDir='/home/eric/sequences_to_contact_maps'


dataset=dataset_01_15_22
dataFolder="${dataDir}/${dataset}"
python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --samples $samples --ref $ref

for sample in 40 1230 1718
do
  python3 ~/TICG-chromatin/scripts/makeLatexTable.py --data_folder $dataFolder --sample $sample --ref $ref
done
