#! /bin/bash

# chi="-1&2&-1&1.5\\2&-1&-1&-0.5\\-1&-1&-1&1.5\\1.5&-0.5&1.5&-1"
# chi="-1&1&0&0\\1&-2&0&-1\\0&0&-1&2\\0&-1&2&-1"
# chi="-1&1\\1&0"
chi='polynomial'
k=4
m=1024
today=$(date +'%m_%d_%y')
dataFolder="/project2/depablo/erschultz/dataset_01_18_22"
startSample=1
relabel='none'
startNode=1
nodes=1
tasks=20
samples=20
diag='false'
nSweeps=1000000
pSwitch=0.05
minChi=-1
maxChi=1
fillDiag='none'
chiSeed='none'
local='false'

if [ $local = 'true' ]
then
  dataFolder="/home/eric/dataset_test"
  scratchDir='/home/eric/scratch'
  startSample=1
  nSweeps=50000
  nodes=1
  tasks=5
  samples=5
  source activate python3.8_pytorch1.8.1_cuda11.1
else
  scratchDir="/scratch/midway2/erschultz"
  source activate python3.8_pytorch1.8.1_cuda10.2
fi

samplesPerNode=$(( $samples / $nodes ))
samplesPerTask=$(( $samplesPerNode / $tasks ))
echo "samples per node" $samplesPerNode
echo "samples per task" $samplesPerTask

cd ~/TICG-chromatin/src
make
mv TICG-engine ..

cd ~/TICG-chromatin

for i in $( seq $startNode $(( $nodes - 1 + $startNode )) )
do
  startSampleI=$(( $startSample + $samplesPerNode * $i ))
  endSampleI=$(( $startSampleI + $samplesPerNode - 1 ))
  echo "TICG${i}" $startSampleI $endSampleI
  sbatch ~/TICG-chromatin/bin/random${i}.sh $chi $k $m $dataFolder $startSampleI $relabel $tasks $samplesPerNode $samplesPerTask $diag $scratchDir $i $nSweeps $pSwitch $minChi $maxChi $fillDiag $chiSeed
done
