#!/bin/bash
##
## Accept input file for graph structure
if [ $# -eq 0 ]; then
  printf "Please include an input file\n"
  exit
else
  INPUT=$1
fi
##
## Compile c++ dijkstra's and a*
g++ -std=c++11 -o find_path find_path.cpp
## 
## Run exe to generate path YAMLs
./find_path $1
##
## Delete generated exe
rm find_path
