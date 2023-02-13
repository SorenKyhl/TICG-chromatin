#! /bin/bash

source ~/TICG-chromatin/bin/random/random_fns.sh
source activate python3.9_pytorch1.9

cd ~/TICG-chromatin

run()  {
	# move utils to scratch
	scratchDirI="${scratchDir}/${i}"
	move

	check_dir

	random_inner

	# clean up
	rm -f default_config.json *.xyz
	rm -d $scratchDirI
}

param_setup
m=1024
dataset=dataset_test
baseDataFolder="/home/erschultz/${dataset}"
dataFolder=$baseDataFolder
scratchDir='/home/erschultz/scratch'
overwrite=1

k=0
nSweeps=5000
dumpFrequency=100
TICGSeed=10
chiSeed=3
seqSeed=4
diag='true'
dense='false'
diagBins=1024
nSmallBins=64
smallBinSize=1
bigBinSize=-1
nBigBins=-1
bondLength=28
useD='true'
useL='true'
useE='true'


sample=3001
chiDiagMethod="zeros"
maxDiagChi=10
chiMethod="none"
seqMethod="none"
lmbda=0.85
i="${sample}"
# run

sample=3002
m=512
diagBins=512
maxDiagChi=5
# seqMethod="${dataFolder}/samples/sample3001/x.npy"
i="${sample}"
# run &

# gaussian renorm
sample=3003
seqMethod="random"
scaleResolution=2
maxDiagChi=20
beadVol=1040
bondLength=39.598
i="${sample}"
# run &

# reset
beadVol=520
bondLength=28
scaleResolution=1

sample=3004
maxDiagChi=5
# seqMethod="${dataFolder}/samples/sample3001/x.npy"
gridSize=28.5
i="${sample}"
# run &

sample=3005
# seqMethod="${dataFolder}/samples/sample3001/x.npy"
gridSize=28
i="${sample}"
# run &

sample=3006
# seqMethod="${dataFolder}/samples/sample3001/x.npy"
gridSize=27.5
i="${sample}"
# run &

sample=3007
# seqMethod="${dataFolder}/samples/sample3001/x.npy"
gridSize=27
i="${sample}"
# run &

sample=3008
# seqMethod="${dataFolder}/samples/sample3001/x.npy"
gridSize=27
phiChromatin=0.03
i="${sample}"
# run &

# reset
gridSize=28.7
phiChromatin=0.06


# attempt with diag params
sample=4001
m=1024
chiDiagMethod="linear"
maxDiagChi=10
# seqMethod="random"
lmbda=0.85
i="${sample}"
run &

sample=4002
m=512
diagBins=512
maxDiagChi=5
# seqMethod="${dataFolder}/samples/sample3001/x.npy"
i="${sample}"
run &

# gaussian renorm
sample=4003
# seqMethod="random"
scaleResolution=2
maxDiagChi=20
beadVol=1040
bondLength=39.598
i="${sample}"
run &

# reset
beadVol=520
bondLength=28
scaleResolution=1

sample=4006
maxDiagChi=5
# seqMethod="${dataFolder}/samples/sample3001/x.npy"
gridSize=27.5
i="${sample}"
run &

sample=4007
maxDiagChi=5
# seqMethod="${dataFolder}/samples/sample3001/x.npy"
gridSize=27
i="${sample}"
run &

sample=4008
# seqMethod="${dataFolder}/samples/sample3001/x.npy"
gridSize=27
phiChromatin=0.03
i="${sample}"
run &


wait
