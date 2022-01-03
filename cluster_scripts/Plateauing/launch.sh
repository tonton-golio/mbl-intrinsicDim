#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/paper_data/plateauing/"

#L=8
#for i in `seq 0 10`
#do
#	qsub -l walltime=00:10:00,mem=1gb -v output_path=$savedir,L=$L,S=$i -N Plateauing-L-$L-S-$i controlscript.pbs
#done
#
#L=10
#for i in `seq 0 10`
#do
#	qsub -l walltime=01:00:00,mem=2gb -v output_path=$savedir,L=$L,S=$i -N Plateauing-L-$L-S-$i controlscript.pbs
#done
#
#L=12
#for i in `seq 0 10`
#do
#	qsub -l walltime=48:00:00,mem=16gb -v output_path=$savedir,L=$L,S=$i -N Plateauing-L-$L-S-$i controlscript.pbs
#done
#
#L=14
#for i in `seq 0 10`
#do
#	qsub -l walltime=96:00:00,mem=32gb -v output_path=$savedir,L=$L,S=$i -N Plateauing-L-$L-S-$i controlscript.pbs
#done

L=16
for i in `seq 0 10`
do
	qsub -l walltime=192:00:00,mem=32gb -v output_path=$savedir,L=$L,S=$i -N Plateauing-L-$L-S-$i controlscript.pbs
done

