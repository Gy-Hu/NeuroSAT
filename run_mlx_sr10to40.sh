#!/bin/sh

python src/train_mlx.py \
  --task-name 'neurosat_mlx' \
  --dim 128 \
  --n_rounds 26 \
  --epochs 200 \
  --n_pairs 100000 \
  --max_nodes_per_batch 12000 \
  --gen_log 'log/data_maker_mlx_sr10t40.log' \
  --min_n 10 \
  --max_n 40 \
  --val-file 'val_v40_vpb12000_b2604.pkl'