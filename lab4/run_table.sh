#!/bin/bash

echo "Processes   Time(Tp)" > results.txt

for p in 1 2 4 8
do
    output=$(mpirun --oversubscribe -np $p ./dot_product)
    echo "$output" >> results.txt
done

echo "Done. Results saved in results.txt"

