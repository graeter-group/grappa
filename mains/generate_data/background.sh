#!/bin/bash

# Generate a random 10-digit number
RANDOM_SUFFIX=$(cat /dev/urandom | tr -dc '0-9' | fold -w 10 | head -n 1)

mkdir -p out

# Build the output file name
OUTPUT_FILE="out/${RANDOM_SUFFIX}.out"

echo executing "$@" and writing output to $OUTPUT_FILE

# Run the program in the background
nohup "$@" > $OUTPUT_FILE 2>&1 &

echo "Your program has been started in the background. Output will be written to $OUTPUT_FILE."
