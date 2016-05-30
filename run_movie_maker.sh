#!/bin/bash

ffmpeg -f image2 -start_number 07 -r 8 -i $1 -vcodec mpeg4 -y $2