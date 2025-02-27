#!/bin/sh

#PBS -N neurosat_sr3t10
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /home/Users/apple/coding_env/NeuroSAT/projects/neurosat/pytorch_neurosat/

python src/train.py \
  --task-name 'neurosat_No2' \
  --dim 128 \
  --n_rounds 26 \
  --epochs 10 \
  --n_pairs 100000 \
  --max_nodes_per_batch 12000 \
  --gen_log '/Users/apple/coding_env/NeuroSAT/log/data_maker_sr3t10.log' \
  --min_n 3 \
  --max_n 10 \
  --data-dir '/Users/apple/coding_env/NeuroSAT/data' \
  --log-dir '/Users/apple/coding_env/NeuroSAT/log' \
  --model-dir '/Users/apple/coding_env/NeuroSAT/model' \
  --val-file 'val_v10_vpb12000_b148.pkl'
