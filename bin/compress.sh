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
  for i in {1..10000}
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

to_small(){
  dataset=$1
  cd $dir
  small_dataset="${dataset}-small"
  mkdir $small_dataset
  cd $small_dataset
  mkdir samples
  for i in {1000..2000}
  do
    cd "${dir}/${dataset}/samples"
    cp -r "sample${i}" "${dir}/${small_dataset}/samples"
  done

  cd $dir
  tar -czvf "${dataset}.tar.gz" $small_dataset
}

dir='/home/erschultz'

dir='/project2/depablo/erschultz'
cd $dir
compress dataset_12_05_22
compress dataset_11_18_22
compress dataset_11_21_22
