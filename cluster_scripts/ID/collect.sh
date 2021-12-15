#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/paper_data/"

#mkdir -p $savedir

module load tools
module load anaconda3/2020.07

L=16
for i in `seq 0 30`
do
#	qsub -l walltime=48:00:00,mem=32gb -v output_path=$savedir,L=$L,S=$i -N collect-L-$L-S-$i controlscript_collect.pbs
	python collect_for_W.py $savedir -L 16 -S $i
done
