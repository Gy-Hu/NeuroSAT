#!/bin/sh

#PBS -N neurosat_sr40t100
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /Users/apple/coding_env/NeuroSAT/

python src/train.py \
  --task-name 'neurosat_3rd_rnd' \
  --dim 128 \
  --n_rounds 32 \
  --epochs 200 \
  --n_pairs 100000 \
  --max_nodes_per_batch 12000 \
  --gen_log '/Users/apple/coding_env/NeuroSAT/log/data_maker_sr40t100.log' \
  --min_n 40 \
  --max_n 100 \
  --restore '/Users/apple/coding_env/NeuroSAT/model/neurosat_2nd_rnd_sr40to100_ep200_nr32_d128_last.pth.tar' \
  --val-file 'val_v100_vpb12000_b1284.pkl'
