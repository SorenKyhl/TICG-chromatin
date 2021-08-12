#!/bin/bash

# this example shows a maximum entropy procedure which
# matches the parameters of a "ground truth" simulation (iteration 0),
# the ground truth chi parameters are specified by the first line of 
# chis.txt and chis_diag.txt:
# 
# chis.txt (line 1):  -1 1 -1
# chis_diag.txt (line 1):  np.linspace(0,0.1,16)
#
# the maximum entropy procedure begins in iteration1,
# with initial chi paramters specified by the second line of 
# chis.txt and chis_diag.txt
#
# chis.txt (line 2): 0 0 0
# chis_diag.txt (line 2): np.zeros(16)

# using all default parameters:
./bin/run.sh 


