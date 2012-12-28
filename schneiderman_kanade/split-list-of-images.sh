#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 file [prefix]"
    exit 1
fi

INPUT=$1
PREFIX=$2

awk -v PREFIX=$PREFIX 'BEGIN{ln=1;}{
  fname = sprintf("%s%d",PREFIX,ln); 
  print $0 > fname;
  close(fname);
  ++ln;
}' $1
