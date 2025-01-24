#!/bin/bash
#$ -cwd
#$ -N L24
#$ -q cpu.q
#$ -pe smp 30
#$ -l h_vmem=128G
#$ -l h_rt=1:00:00
#$ -o /ddn/niknovikov19/repo/model_tuner/models/L24/exp_logs/L24_log
#$ -e /ddn/niknovikov19/repo/model_tuner/models/L24/exp_logs/L24_err

source ~/.bashrc
#echo $(pwd)
conda activate netpyne_batch
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /ddn/niknovikov19/repo/model_tuner/models/L24
mpiexec -n $NSLOTS -hosts $(hostname) nrniv -python -mpi init.py
