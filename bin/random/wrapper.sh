#! /bin/bash

# chi="-1&2&-1&1.5\\2&-1&-1&-0.5\\-1&-1&-1&1.5\\1.5&-0.5&1.5&-1"
# chi="-1&1&0&0\\1&-2&0&-1\\0&0&-1&2\\0&-1&2&-1"
# chi="0&0\\0&0"
m=1024
today=$(date +'%m_%d_%y')
dataFolder="/project2/depablo/dataset_test"
startSample=4
relabel='none'
startNode=0
nodes=8
tasks=20
samples=1600
nSweeps=1000000
lmbda=0.85

# plaid
k=4
chiMethod='random'
minChi=-2
maxChi=2
chiSeed='none'
fillDiag='none'

# diag
diagMethod='linear'
maxDiagChi=4
dense='true'
diagBins=76
smallBinSize=4
bigBinSize=19
nSmallBins=28
nBigBins=48

local='true'

if [ $local = 'true' ]
then
  dataFolder="/home/erschultz/dataset_test"
  scratchDir='/home/erschultz/scratch'
  startSample=4
  nSweeps=1000000
  nodes=1
  tasks=3
  samples=3
  source activate python3.9_pytorch1.9
else
  scratchDir="/scratch/midway2/erschultz"
  source activate python3.9_pytorch1.9_cuda10.2
fi

samplesPerNode=$(( $samples / $nodes ))
samplesPerTask=$(( $samplesPerNode / $tasks ))
echo "samples per node" $samplesPerNode
echo "samples per task" $samplesPerTask

# cd ~/TICG-chromatin/src
# make
# mv TICG-engine ..

cd ~/TICG-chromatin

for i in $( seq $startNode $(( $nodes - 1 + $startNode )) )
do
  startSampleI=$(( $startSample + $samplesPerNode * $i ))
  endSampleI=$(( $startSampleI + $samplesPerNode - 1 ))
  echo "TICG${i}" $startSampleI $endSampleI "m=${m}"
  bash ~/TICG-chromatin/bin/random/random${i}.sh $chiMethod $k $m $dataFolder $startSampleI $relabel $tasks $samplesPerNode $samplesPerTask $diagMethod $scratchDir $i $nSweeps $lmbda $minChi $maxChi $fillDiag $chiSeed $maxDiagChi $dense $diagBins $smallBinSize $bigBinSize $nSmallBins $nBigBins
done
