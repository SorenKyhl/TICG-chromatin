#! /bin/bash
#SBATCH --job-name=compress
#SBATCH --output=logFiles/compress.out
#SBATCH --time=24:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=1000

compress(){
  dataset=$1
  echo $dataset
  for i in {1..10000}
  do
    cd "${dir}/${dataset}/samples/sample${i}"
    # energy
    rm e.npy
    rm s.npy
    rm L.npy
    # rm S.npy

    rm chis.tek
    rm chis.npy
    rm *diag.npy
    rm *.png

    rm *.txt

    rm -r equilibration


    cd production_out
    rm *.traj
  done

  # cd $dir
  # echo $dir
  # rm -r "${dataset}.tar.gz"
  # tar -czf "${dataset}.tar.gz" $dataset
  # rm -r $dataset
}

compress2(){
    dataset=$1
    echo $dataset

    cd $dir
    echo $dir
    rm -r "${dataset}.tar.gz"
    tar -czf "${dataset}.tar.gz" $dataset
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

cleanup(){
  dataset=$1
  echo $dataset
  for i in {1..10000}
  do
    cd "${dir}/${dataset}/samples/sample${i}"
    # energy
    rm e.npy &
    rm s.npy &
    rm *.png &

    wait

    cd production_out
    rm *.traj
  done
}

dir='/home/erschultz'
dir='/project2/depablo/erschultz'
cd $dir
compress dataset_02_30_24_imr90 &
compress dataset_03_12_24_imr90 &
wait
