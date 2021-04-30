#!/bin/bash
##
## CHECKLIST BEFORE RUNNING THIS SCRIPT
##	- may have to run dos2unix visualize.sh
##	- may have to run chmod +x visualize.sh
##	- must have python 3.6+ installed
##	- must have run py -m pip install -U matplotlib
##	- must have run py -m pip install -U pyyaml
##
## HOW TO RUN
## 	./visualize.sh [path to SampleMap]
##
## NOTICE:
## 	If the visualizer is exited before completion,
## 	there is chance that the code will continue running.
## 	In this case, use ctrl+c and then run
## 	cleanup.py [path to SampleMap]
## 	to remove generated files
##
## Accept input file for graph structure
if [ $# -eq 0 ]; then
  printf "Please include a path to an input file\n"
  exit
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
##
## Run visualizer on file
./visualizer.exe $1
##
## Clean up generated files
python3 cleanup.py $1
