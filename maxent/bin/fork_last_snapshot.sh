#!/bin/bash
# takes last snapshot of output.xyz and saves it in the desired file

ROWS=`head -1 data_out/output.xyz`
let ROWS++
let ROWS++
tail -$ROWS data_out/output.xyz > $1

