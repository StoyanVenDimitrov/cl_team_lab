#!/bin/bash

for i in $(ls configs/torch/*)
do
 CUDA_VISIBLE_DEVICES=1 python3 main_torch.py --train True --config $i
done
