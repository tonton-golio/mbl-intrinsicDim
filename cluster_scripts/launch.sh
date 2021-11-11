#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/paper_data/"

for L in 8 #10 12 14 16
do
	for i in 0 #`seq 0 9`
	do
		qsub -v output_path=$savedir,L=$L,S=$i, -N 2NN-L-$L-S-$i controlscript.pbs
	done
done
