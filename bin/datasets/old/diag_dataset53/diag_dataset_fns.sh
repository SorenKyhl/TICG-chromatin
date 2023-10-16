#! /bin/bash

source ~/TICG-chromatin/bin/datasets/dataset_fns.sh

dataset=dataset_09_19_23
dataFolder="/project2/depablo/erschultz/${dataset}"
scratchDir='/home/erschultz/scratch'
#
# dataFolder="/home/erschultz/${dataset}"
# scratchDir='/home/erschultz/scratch'

m=512
overwrite=1
useL='true'
useS='true'
useD='true'

nSweeps=500000
dumpFrequency=100000
TICGSeed=10
dense='false'
diagBins=512
phiChromatin=0.01
beadVol=130000
bondLength=180
maxDiagChi='none'
trackContactMap='true'
