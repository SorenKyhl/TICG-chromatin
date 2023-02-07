#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --output=logFiles/cleanup2.out
#SBATCH --time=5:00:00
#SBATCH --partition=depablo
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

dir='/project2/depablo/erschultz'
dataset=dataset_09_30_22
for i in {21..2520}
do
  cd "${dir}/${dataset}/samples/sample${i}"
  # max ent
  rm -r GNN* &
  rm -r ground* &
  rm -r k_means* &
  rm -r PCA* &
  rm -r random* &

  # energy
  rm e.npy &
  rm s.npy &

  rm -r data_out &
  rm chis.tek &
  rm chis.npy &
  rm *diag.npy &
  rm *.png &

  wait
done

cd $dir
rm -r "${dataset}.tar.gz"
tar -czvf "${dataset}.tar.gz" $dataset
rm -r $dataset
