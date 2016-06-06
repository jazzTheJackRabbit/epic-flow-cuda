#!/bin/bash
# matlab -nodesktop -nojvm -r "addpath('/Users/amogh/Documents/MATLAB/structured_edge_detector_lib'); addpath(genpath('/Users/amogh/Documents/MATLAB/pdollar_cv_toolbox')); load('modelFinal.mat'); I = imread('$1'); if size(I,3)==1, I = cat(3,I,I,I); end; edges = edgesDetect(I, model); fid=fopen('$3','wb'); fwrite(fid,transpose(edges),'single'); fclose(fid); exit"

# ../deep_matching/deepmatching_c++/deepmatching $1 $2 -png_settings -out $4

build/epic_flow $1 $2 $3 $4 $5 $7 $8 $9 

../flow-code/color_flow $5 $6