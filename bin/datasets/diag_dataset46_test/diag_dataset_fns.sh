#! /bin/bash

source ~/TICG-chromatin/bin/datasets/dataset_fns.sh

dataset=dataset_08_28_23
# dataFolder="/project2/depablo/erschultz/${dataset}"
# scratchDir='/home/erschultz/scratch'
#
dataFolder="/home/erschultz/${dataset}"
scratchDir='/home/erschultz/scratch'

useL='true'
useS='true'
useD='true'

m=512
nSweeps=5000
dumpFrequency=1000
TICGSeed=10
dense='false'
diagBins=512
phiChromatin=0.01
beadVol=130000
bondLength=261
maxDiagChi='none'
trackContactMap='true'
