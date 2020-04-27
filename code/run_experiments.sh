#!/bin/bash

cd $1

for file in $(ls ./) 
do
    if [[ $file == exp* ]]
    then
        echo "Issuing: ./$file > res_$file.txt 2>&1"
        ./$file > res_$file.txt 2>&1 &
    fi
done

echo "Waiting..."
wait

exit
