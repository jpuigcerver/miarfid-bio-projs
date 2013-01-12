#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 training_data"
    exit 1
fi

REG_W=(4 5 6 7)
REG_H=(4 5 6 7)

STP_X=(1 2 3 4 5 6 7)
STP_Y=(1 2 3 4 5 6 7)

# Find the best region size and step size
if [ ${#REG_W[@]} -ne ${#REG_H[@]} ]; then
    echo "Arrays REG_W and REG_H must have the same number of elements."
    exit 1
fi
if [ ${#STP_X[@]} -ne ${#STP_Y[@]} ]; then
    echo "Arrays STP_X and STP_Y must have the same number of elements."
    exit 1
fi

reg_i=0
while [ $reg_i -lt ${#REG_H[@]} ]; do
    reg_w=${REG_W[$reg_i]}
    reg_h=${REG_H[$reg_i]}
    stp_i=0
    while [ $stp_i -lt ${#STP_X[@]} ]; do
	stp_x=${STP_X[$stp_i]}
	stp_y=${STP_Y[$stp_i]}
	d=$[reg_w * reg_h]
	echo $reg_w $reg_h $stp_x $stp_y $d
	rep=1
	while [ $rep -le $NUM_REPS ]; do
	    rep=$[rep + 1]
	done
	stp_i=$[stp_i + 1]
    done
    reg_i=$[reg_i + 1]
done