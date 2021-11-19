#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/paper_data/"

L = 8
for i in `seq 0 2`
do
	qsub -l walltime=00:10:00,mem=1gb -v output_path=$savedir,L=$L,S=$i -N 2NN-L-$L-S-$i controlscript.pbs
done

L = 10
for i in `seq 0 2`
do
	qsub -l walltime=01:00:00,mem=2gb -v output_path=$savedir,L=$L,S=$i -N 2NN-L-$L-S-$i controlscript.pbs
done

L = 12
for i in `seq 0 2`
do
	qsub -l walltime=04:00:00,mem=4gb -v output_path=$savedir,L=$L,S=$i -N 2NN-L-$L-S-$i controlscript.pbs
done

L = 14
for i in `seq 0 2`
do
	qsub -l walltime=24:00:00,mem=8gb -v output_path=$savedir,L=$L,S=$i -N 2NN-L-$L-S-$i controlscript.pbs
done

L = 16
for i in `seq 0 2`
do
	qsub -l walltime=48:00:00,mem=10gb -v output_path=$savedir,L=$L,S=$i -N 2NN-L-$L-S-$i controlscript.pbs
done

