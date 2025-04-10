#!/bin/bash
#$ -cwd
#$ -N BATCH_MAIN
#$ -q cpu.q
#$ -pe smp 8
#$ -l h_vmem=64G
#$ -V
#$ -l h_rt=12:00:00
#$ -o /ddn/niknovikov19/test/model_tuner/test_opt_L24_batch_qsub/exp_r0_2_10_5_15_pfr_0.1_1.5_7_wmult_0.25_alpha_0.25/log/batch_script_log.out
#$ -e /ddn/niknovikov19/test/model_tuner/test_opt_L24_batch_qsub/exp_r0_2_10_5_15_pfr_0.1_1.5_7_wmult_0.25_alpha_0.25/log/batch_script_log.err

source ~/.bashrc
conda activate netpyne_batch
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /ddn/niknovikov19/repo/model_tuner/models/L24
python opt_batch_script.py "/ddn/niknovikov19/test/model_tuner/test_opt_L24_batch_qsub/exp_r0_2_10_5_15_pfr_0.1_1.5_7_wmult_0.25_alpha_0.25"