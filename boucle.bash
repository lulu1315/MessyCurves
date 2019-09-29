#!/bin/bash

start=51
end=100


for i in `seq $start $end`;
do
    ii=$(printf "%04d" $i)
    j=$(echo ".001 * $i" | bc -l)
    echo $j
    ./messycurves ../images/dianaross1_normalize.png diana.$ii.png 5000 12 10 .2 $j
done    
