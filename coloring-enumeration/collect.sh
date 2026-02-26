#!/bin/zsh

# want to get some data on these distributions.
squares=(4 9 16 25 36)

for square in $squares; do
	./count $square 3 > data/$square.3.csv
done
