#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/paper_data/"

# Use batch for L=8, bc a single run over all Ws takes just a second
#L=8
#for i in `seq 0 1000 100000`
#do
#	Ss=$i
#        let Se=$i+1000
#	qsub -l walltime=01:00:00,mem=1gb -v output_path=$savedir,L=$L,Ss=$Ss,Se=$Se -N 2NN-L-$L-S-$i controlscript_batch.pbs
#done

#L=10
#for i in `seq 0 100 10000`
#do
#	Ss=$i
#        let Se=$i+100
#	qsub -l walltime=02:00:00,mem=2gb -v output_path=$savedir,L=$L,Ss=$Ss,Se=$Se -N 2NN-L-$L-S-$i controlscript_batch.pbs
#done
#

# L = 12 takes ~5 min each, so we can also do multiple in 1 job
#L=12
#for i in `seq 0 10 5000`
#do
#	Ss=$i
#        let Se=$i+10
#	qsub -l walltime=04:00:00,mem=4gb -v output_path=$savedir,L=$L,Ss=$Ss,Se=$Se -N 2NN-L-$L-S-$i controlscript_batch.pbs
#done

#L=14
#for i in `seq 0 1000`
#do
#	qsub -l walltime=24:00:00,mem=8gb -v output_path=$savedir,L=$L,S=$i -N 2NN-L-$L-S-$i controlscript.pbs
#done
#
L=16
for i in `seq 500 999`
do
	for W in 1.   1.2  1.4  1.6  1.8  2.   2.2  2.4  2.6  2.65 2.7  2.75 2.8  2.85 2.9  2.95 3.   3.05 3.1  3.15 3.2  3.25 3.3  3.35 3.4  3.45 3.5  3.55 3.6  3.65 3.7  3.75 3.8  3.85 3.9  3.95 4.   4.05 4.1  4.15 4.2  4.25 4.3  4.35 4.4  4.45 4.5  4.7  4.9  5.1  5.3  5.5  5.7  5.9  6.1
	do
		qsub -l walltime=48:00:00,mem=32gb -v output_path=$savedir,L=$L,S=$i,W=$W -N 2NN-L-$L-S-$i-W-$W controlscript_for_W.pbs
	done
done

#L=18
#for i in 95 #`seq 0 100`
#do
#	qsub -l walltime=48:00:00,mem=16gb -v output_path=$savedir,L=$L,S=$i -N 2NN-L-$L-S-$i controlscript.pbs
#done
#
