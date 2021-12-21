#! /bin/bash
#SBATCH --job-name=rename
#SBATCH --output=logFiles/rename.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --qos=depablo-debug

# dir="/project2/depablo/erschultz/dataset_11_03_21/samples"
# for i in `seq 40`
# do
#   cd "${dir}/sample${i}"
#   mv x_linear.npy psi.npy
# done
# 
# dir="/project2/depablo/erschultz/dataset_12_11_21/samples"
# for i in `seq 40`
# do
#   cd "${dir}/sample${i}"
#   mv x_linear.npy psi.npy
# done


dir="/project2/depablo/erschultz/dataset_12_17_21/samples"
for i in `seq 40`
do
  cd "${dir}/sample${i}"
  mv x_linear.npy psi.npy
done
