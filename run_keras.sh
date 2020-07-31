#!/bin/bash

for i in $(ls configs/keras/)
do
 python3 main_keras.py --train True --config configs/keras/$i
done
