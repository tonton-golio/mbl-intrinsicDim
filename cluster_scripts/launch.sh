#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/results_paper/"

L=14

for i in `seq 0 9`
do
	qsub -v output_path=$savedir,L=$L,S=$i, -N hubbard-2NN-L-$L-S-$i controlscript.pbs
done
