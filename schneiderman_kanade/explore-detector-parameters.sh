#!/bin/bash

LANG=C
if [ $# -ne 1 ]; then
    echo "Usage: $0 training_data"
    exit 1
fi

REG_W=(4 5 6 7)
REG_H=(4 5 6 7)

STP_X=(1 3 5)
STP_Y=(1 3 5)

MIN_D=5
MAX_D=25
INC_D=5

MIN_K=10
MAX_K=250
INC_K=10

NUM_REPS=10

if [ ${#REG_W[@]} -ne ${#REG_H[@]} ]; then
    echo "Arrays REG_W and REG_H must have the same number of elements."
    exit 1
fi
if [ ${#STP_X[@]} -ne ${#STP_Y[@]} ]; then
    echo "Arrays STP_X and STP_Y must have the same number of elements."
    exit 1
fi

# Returns a random number
function rand_seed {
    cat /dev/urandom | od -N4 -An -t u4 | awk '{print $1}'
}

function run_experiment() {
    img_w=$2
    img_h=$3
    reg_w=$4
    reg_h=$5
    stp_x=$6
    stp_y=$7
    d=$8
    k=$9
    sum_dscore=0
    sum_sq_dscore=0
    sum_time=0
    sum_sq_time=0
    rep=1
    while [ $rep -le $NUM_REPS ]; do
	seed=$(rand_seed)
	base_tmp=/tmp/$(basename $1)-imgw$img_w-imgh$img_h-regw$reg_w-regh$reg_h-d$d-k$k-stpx$stp_x-stpy$stp_y-rep$rep-seed$seed
	train=$base_tmp-train
	valid=$base_tmp-valid
	log=$base_tmp-log
	./partition-dataset.sh $1 $train $valid 0.1
	out=$base_tmp-out
	time=$base_tmp-time
	/usr/bin/time -f '%e' -o $time ./sk-train -img_h $img_h -img_w $img_w \
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
	rm -f $train $valid $log $out $time
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
    echo $avg_dscore $var_dscore $conf_dscore \
	$avg_time $var_time $conf_time
}

# Find the best region size and step size
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
	echo -n "$reg_w $reg_h $stp_x $stp_y "
	run_experiment $1 21 21 $reg_w $reg_h $stp_x $stp_y $d $k
	stp_i=$[stp_i + 1]
    done
    reg_i=$[reg_i + 1]
done

# Find the best dimensionality
echo ""
d=$MIN_D
while [ $d -le $MAX_D ]; do
    echo -n "$d "
    run_experiment $1 21 21 5 5 3 3 $d 50
    d=$[d + INC_D]
done

# Find the best number of quantized patterns
echo ""
k=$MIN_K
while [ $k -le $MAX_K ]; do
    echo -n "$k "
    run_experiment $1 21 21 5 5 3 3 15 $k
    k=$[k + INC_K]
done