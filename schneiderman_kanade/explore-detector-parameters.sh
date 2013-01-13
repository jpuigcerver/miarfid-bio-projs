#!/bin/bash

LANG=C
if [ $# -ne 1 ]; then
    echo "Usage: $0 training_data"
    exit 1
fi

REG_W=(5 6 7)
REG_H=(5 6 7)

STP_X=(1 3 5)
STP_Y=(1 3 5)

NUM_REPS=10

# Returns a random number
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
k=50
d=$[REG_W[0] * REG_H[0]]
while [ $reg_i -lt ${#REG_H[@]} ]; do
    reg_w=${REG_W[$reg_i]}
    reg_h=${REG_H[$reg_i]}
    stp_i=0
    while [ $stp_i -lt ${#STP_X[@]} ]; do
	stp_x=${STP_X[$stp_i]}
	stp_y=${STP_Y[$stp_i]}
	rep=1
	sum_dscore=0
	sum_sq_dscore=0
	sum_time=0
	sum_sq_time=0
	while [ $rep -le $NUM_REPS ]; do
	    seed=$(rand_seed)
	    cfg="regw$reg_w-regh$reg_h-d$d-k$k-stpx$stp_x-stpy$stp_y-rep$rep-seed$seed"
	    train=/tmp/$(basename $1)-train-$cfg
	    valid=/tmp/$(basename $1)-valid-$cfg
	    log=/tmp/$(basename $1)-log-$cfg
	    ./partition-dataset.sh $1 $train $valid 0.1
	    out=/tmp/$(basename $1)-out-$cfg
	    time=/tmp/$(basename $1)-time-$cfg
	    /usr/bin/time -f '%e' -o $time ./sk-train -img_h 21 -img_w 21 \
		-d $d -k $k -reg_h $reg_h -reg_w $reg_w -seed $seed \
		-stp_x $stp_x -stp_y $stp_y -train $train -valid $valid \
		-logtostderr 2> $log > $out
	    if [ $? -ne 0 ]; then
		echo "Error. Take a look at log $log"
		exit 1
	    fi
	    dscore=$(cat $out | awk '{print $4}')
	    sum_dscore=$(echo $dscore + $sum_dscore | bc -l)
	    sum_sq_dscore=$(echo "( $dscore * $dscore ) + $sum_sq_dscore" | \
		bc -l)
	    time=$(cat $time)
	    sum_time=$(echo $time + $sum_time | bc -l)
	    sum_sq_time=$(echo "( $time * $time ) + $sum_sq_time" | bc -l)
	    rm -f $train $valid $log $out
	    rep=$[rep + 1]
	done
	avg_dscore=$(echo $sum_dscore / $NUM_REPS | bc -l)
	var_dscore=$( \
	    echo "($sum_sq_dscore / $NUM_REPS) - ($avg_dscore * $avg_dscore)" | \
	    bc -l)
	conf_dscore=$(echo "1.96 * sqrt($var_dscore / $NUM_REPS)" | bc -l)
	avg_time=$(echo "$sum_time / $NUM_REPS" | bc -l)
	var_time=$( \
	    echo "($sum_sq_time / $NUM_REPS) - ($avg_time * $avg_time)" | \
	    bc -l)
	conf_time=$(echo "1.96 * sqrt($var_time / $NUM_REPS)" | bc -l)
	echo $reg_w $reg_h $stp_x $stp_y $avg_dscore $var_dscore $conf_dscore \
	    $avg_time $var_time $conf_time
	stp_i=$[stp_i + 1]
    done
    reg_i=$[reg_i + 1]
done