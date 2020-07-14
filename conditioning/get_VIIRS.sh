#!/bin/sh

for line in $(wget -q -nH -nd $1 -O - | grep .tar\" | cut -f2 -d\")
do
	wget $line -P $2
done

# TO UNPACK
#python unpack_data.py $2
