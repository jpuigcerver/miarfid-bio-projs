#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Usage: $0 input part1 part2 [f]"
    exit 1
fi
f=$4; if [ $# -lt 4 ]; then f=0.8; fi

shuf_input=/tmp/$(basename $1).shuf.$RANDOM
shuf --random-source=/dev/urandom $1 > $shuf_input
total_lines=$(cat $shuf_input | wc -l)
part1_lines=$(python -c "print int($total_lines * $f)")
part2_lines=$[total_lines - part1_lines]
head -n $part1_lines $shuf_input > $2
tail -n $part2_lines $shuf_input > $3

rm -f $shuf_input