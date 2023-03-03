#! /bin/bash
#SBATCH --job-name=compress
#SBATCH --output=logFiles/compress.out
#SBATCH --time=24:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1000

compress(){
  dataset=$1
  for i in {1..5000}
  do
    cd "${dir}/${dataset}/samples/sample${i}"
    # energy
    rm e.npy &
    rm s.npy &

    rm -r data_out &
    rm chis.tek &
    rm chis.npy &
    rm *diag.npy &
    rm *.png &

    rm *.txt &

    wait
  done

  cd $dir
  rm -r "${dataset}.tar.gz"
  tar -czvf "${dataset}.tar.gz" $dataset
  rm -r $dataset
}


dir='/project2/depablo/erschultz'
compress dataset_02_27_23
