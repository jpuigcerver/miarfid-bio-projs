#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 faces_prefix nfaces_prefix"
    exit 1
fi

function dirname () {
  echo $1 | awk -F '/' '{
    $NF=""; d=$1; 
    for(i=2;i<NF;++i){
      d = sprintf("%s/%s",d,$i);
    } 
    print d
  }'
}

function basename () {
  echo $1 | awk -F '/' '{print $NF}'
}


FACES_BASENAME=$(basename $1)
NFACES_BASENAME=$(basename $2)
FACES_DIRNAME=$(dirname $1)
NFACES_DIRNAME=$(dirname $2)

if [ $FACES_DIRNAME == "" ]; then
    FACES_DIRNAME="."
fi

if [ $NFACES_DIRNAME == "" ]; then
    NFACES_DIRNAME="."
fi

if [ ! -d $FACES_DIRNAME ]; then
    echo "$FACES_DIRNAME is not a directory."
    exit 1
fi

if [ ! -d $NFACES_DIRNAME ]; then
    echo "$NFACES_DIRNAME is not a directory."
    exit 1
fi

for f in $(find $NFACES_DIRNAME -name "${NFACES_BASENAME}*" -type f); do
    echo $f 0
done

for f in $(find $FACES_DIRNAME -name "${FACES_BASENAME}*" -type f); do
    echo $f 1
done
