#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/paper_data/"

# Use batch for L=8, bc a single run over all Ws takes just a second
L=8
for i in `seq 0 100 100000`
do
	Ss=$i
        let Se=$i+100
	qsub -l walltime=00:10:00,mem=1gb -v output_path=$savedir,L=$L,Ss=$Ss,Se=$Se -N 2NN-L-$L-S-$i controlscript_batch.pbs
done

L=10
for i in `seq 0 100 100000`
do
	Ss=$i
        let Se=$i+100
	qsub -l walltime=01:00:00,mem=2gb -v output_path=$savedir,L=$L,Ss=$Ss,Se=$Se -N 2NN-L-$L-S-$i controlscript_batch.pbs
done

# L = 12 takes ~5 min each, so we can also do multiple in 1 job
L=12
for i in `seq 0 10 10000`
do
	Ss=$i
        let Se=$i+10
	qsub -l walltime=04:00:00,mem=4gb -v output_path=$savedir,L=$L,Ss=$Ss,Se=$Se -N 2NN-L-$L-S-$i controlscript_batch.pbs
done

L=14
for i in `seq 0 2`
do
	qsub -l walltime=24:00:00,mem=8gb -v output_path=$savedir,L=$L,S=$i -N 2NN-L-$L-S-$i controlscript.pbs
done

L=16
for i in `seq 0 2`
do
	qsub -l walltime=48:00:00,mem=10gb -v output_path=$savedir,L=$L,S=$i -N 2NN-L-$L-S-$i controlscript.pbs
done
