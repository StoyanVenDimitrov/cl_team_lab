#!/bin/bash

mkdir data && cd data
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz
tar -xzf scicite.tar.gz
rm -rf scicite.tar.gz
