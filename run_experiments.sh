#!/bin/bash

python3 main.py --train True --config configs/conf1
sleep 5 &
python3 main.py --train True --config configs/conf2
sleep 5 &
python3 main.py --train True --config configs/conf3
sleep 5 &
python3 main.py --train True --config configs/conf4

