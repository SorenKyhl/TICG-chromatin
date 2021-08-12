#!/bin/bash
# creates random chi init list with chis between min and max
# usage: $random min max

min=$1
max=$2

function rng {
    echo $((min + $RANDOM % (max+1 - min)))
}

function rand_float {
# generates random number with one floating point digit
    tenths=$(( $RANDOM % 10 ))
    echo "$(rng).$tenths"
}

echo "$(rand_float) $(rand_float) $(rand_float)" > chis.txt
echo "$(rand_float) $(rand_float) $(rand_float)" >> chis.txt
