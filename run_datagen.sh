#!/bin/bash

SAMPLES_PER_CHUNK=100
TOTAL_TRAIN_SAMPLES=1000
TOTAL_TEST_SAMPLES=100

if [ ! -f "results/best_ho_geometry.npy" ]; then
    python main.py opt
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

NUM_TRAIN_CHUNKS=$(( (TOTAL_TRAIN_SAMPLES + SAMPLES_PER_CHUNK - 1) / SAMPLES_PER_CHUNK ))

for (( i=0; i<$NUM_TRAIN_CHUNKS; i++ ))
do
    START_INDEX=$(( i * SAMPLES_PER_CHUNK ))
    python main.py datagen --data-type train --num-samples $SAMPLES_PER_CHUNK --start-index $START_INDEX
    if [ $? -ne 0 ]; then
        exit 1
    fi
done

NUM_TEST_CHUNKS=$(( (TOTAL_TEST_SAMPLES + SAMPLES_PER_CHUNK - 1) / SAMPLES_PER_CHUNK ))

for (( i=0; i<$NUM_TEST_CHUNKS; i++ ))
do
    START_INDEX=$(( i * SAMPLES_PER_CHUNK ))
    python main.py datagen --data-type test --num-samples $SAMPLES_PER_CHUNK --start-index $START_INDEX
    if [ $? -ne 0 ]; then
        exit 1
    fi
done
