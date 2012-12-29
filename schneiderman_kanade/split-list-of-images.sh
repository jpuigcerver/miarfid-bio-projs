#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 file [prefix]"
    exit 1
fi

INPUT=$1
PREFIX=$2

function dirname () {
  echo $1 | awk -F '/' '{
    $NF=""; d=$1;
    for(i=2;i<NF;++i){
      d = sprintf("%s/%s",d,$i);
    }
    print d
  }'
}

PREFIX_BASEDIR=$(dirname $PREFIX)
if [ -n $PREFIX_BASEDIR ]; then
    mkdir -p $PREFIX_BASEDIR
fi

awk -v PREFIX=$PREFIX 'BEGIN{ln=1;}{
  fname = sprintf("%s%d",PREFIX,ln); 
  print $0 > fname;
  close(fname);
  ++ln;
}' $1
