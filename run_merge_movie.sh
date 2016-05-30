#!/bin/bash

cd $1

ffmpeg -i original-movie.mp4 -i flow-movie.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" output.mp4

for i in {1..5}; do printf "file '%s'\n" output.mp4 >> list.txt; done

ffmpeg -f concat -i list.txt -c copy output-final.mp4

rm list.txt
rm output.mp4