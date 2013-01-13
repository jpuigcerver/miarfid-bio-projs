#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 training_data"
    exit 1
fi

REG_W=(5 6 7)
REG_H=(5 6 7)

STP_X=(1 2 3 4 5 6 7)
STP_Y=(1 2 3 4 5 6 7)

NUM_REPS=1

function rand_seed {
    cat /dev/urandom | od -N4 -An -t u4 | awk '{print $1}'
}


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
k=100
d=10
while [ $reg_i -lt ${#REG_H[@]} ]; do
    reg_w=${REG_W[$reg_i]}
    reg_h=${REG_H[$reg_i]}
    stp_i=0
    while [ $stp_i -lt ${#STP_X[@]} ]; do
	stp_x=${STP_X[$stp_i]}
	stp_y=${STP_Y[$stp_i]}
	rep=1
	sum_err=0
	while [ $rep -le $NUM_REPS ]; do
	    seed=$(rand_seed)
	    cfg="regw$reg_w-regh$reg_h-d$d-k$k-stpx$stp_x-stpy$stp_y-seed$seed"
	    train=/tmp/$(basename $1)-train-$cfg
	    valid=/tmp/$(basename $1)-valid-$cfg
	    log=/tmp/$(basename $1)-log-$cfg
	    ./partition-dataset.sh $1 $train $valid 0.8
	    out=$(./sk-train -img_h 21 -img_w 21 -d $d -k $k \
		-reg_h $reg_h -reg_w $reg_w -seed $seed \
		-stp_x $stp_x -stp_y $stp_y -train $train -valid $valid \
		-logtostderr 2> $log)
	    if [ $? -ne 0 ]; then
		echo "Error. Take a look at log $log"
		exit 1
	    fi
	    err=$(echo $out | awk '{print $4}')
	    sum_err=$(echo $err + $sum_err | bc -l)
	    rm -f $train $valid $log
	    rep=$[rep + 1]
	done
	avg_err=$(echo $sum_err / $NUM_REPS | bc -l)
	echo $reg_w $reg_h $stp_x $stp_y $avg_err
	stp_i=$[stp_i + 1]
    done
    reg_i=$[reg_i + 1]
done