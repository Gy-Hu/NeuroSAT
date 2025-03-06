#!/bin/sh

python src/eval_mlx.py \
  --task-name 'neurosat_mlx_eval_sr40' \
  --dim 128 \
  --n_rounds 1024 \
  --restore 'model/neurosat_mlx_sr10to40_ep200_nr26_d128_best.npz' \
  --data-dir 'data/eval/40/'