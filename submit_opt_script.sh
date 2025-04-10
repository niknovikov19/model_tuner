#!/bin/bash
#$ -N OPT
#$ -pe smp 8
#$ -l h_vmem=32G
#$ -o /ddn/niknovikov19/repo/model_tuner/opt_script.run

cd /ddn/niknovikov19/repo/model_tuner
conda activate netpyne_batch
python tests/opt/opt_A1_hpc_batch_qsub/test_opt_A1_hpc_batch_qsub.py
