#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logFiles/cleanup.out
#SBATCH --time=6:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

dir=/project2/depablo/erschultz

cd $dir

tar -xzvf dataset_09_30_22.tar.gz dataset_09_30_22_mini

for i in {1..20}
do
  mv "dataset_09_30_22_mini/samples/sample${i}" "dataset_09_30_22/samples/sample${i}"
done
