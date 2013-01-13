#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 faces nfaces"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "$1 does not exist."
    exit 1
fi

if [ ! -f $2 ]; then
    echo "$2 does not exist."
    exit 1
fi

awk '{print 1, $0}' $1
awk '{print 0, $0}' $2