#!/bin/bash

mkdir data && cd data
wget -O stopwords_en.txt "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz
tar -xzf scicite.tar.gz
rm -rf scicite.tar.gz

