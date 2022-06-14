#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/paper_data/"

#mkdir -p $savedir

module load tools
module load anaconda3/2020.07

cd $PBS_O_WORKDIR

#L=16
#for i in `seq 0 999`
#do
#	python collect_for_W.py $savedir -L $L -S $i
#done


L=18
for i in `seq 0 99`
do
	python collect_for_W.py $savedir -L $L -S $i
done
