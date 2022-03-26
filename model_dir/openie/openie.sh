#!/bin/bash

src=$1
tgt=$2
mem=$3

for file in `find ${src} -name "*.jar"`; do
    if [ -z "${CLASSPATH}" ]
    then
        CLASSPATH="`realpath $file`"
    else
        CLASSPATH="$CLASSPATH:`realpath $file`"
    fi
done

CLASSPATH=$CLASSPATH java -mx${mem} edu.stanford.nlp.naturalli.OpenIE -exec.verbose -ssplit.newlineIsSentenceBreak always -format ollie -filelist ${tgt}

