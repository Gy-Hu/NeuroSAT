#!/bin/sh

#PBS -N neurosat_sr200t500
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /Users/apple/coding_env/NeuroSAT/

python src/train.py \
  --task-name 'neurosat' \
  --dim 128 \
  --n_rounds 64 \
  --epochs 500 \
  --n_pairs 50000 \
  --max_nodes_per_batch 15000 \
  --gen_log '/Users/apple/coding_env/NeuroSAT/log/data_maker_sr200t500.log' \
  --min_n 200 \
  --max_n 500 \
  --val-file 'val_v500_vpb15000_b2564.pkl'
