#! /bin/bash

source ~/TICG-chromatin/bin/datasets/dataset_fns.sh

dataset=dataset_09_29_23_test
# dataFolder="/project2/depablo/erschultz/${dataset}"
# scratchDir='/home/erschultz/scratch'
#
dataFolder="/home/erschultz/${dataset}"
scratchDir='/home/erschultz/scratch'

m=512
useL='true'
useS='true'
useD='true'

nSweeps=4000
dumpFrequency=1000
TICGSeed=10
dense='false'
diagBins=512
phiChromatin=0.008
beadVol=130000
bondLength=180
maxDiagChi='none'
trackContactMap='true'
