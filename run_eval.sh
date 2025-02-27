#!/bin/sh

#PBS -N neurosat_eval_sr40
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /Users/apple/coding_env/NeuroSAT/

python src/eval.py \
  --task-name 'neurosat_eval_sr40' \
  --dim 128 \
  --n_rounds 1024 \
  --restore '/Users/apple/coding_env/NeuroSAT/model/neurosat_3rd_rnd_sr10to40_ep200_nr26_d128.pth.tar' \
  --data-dir '/Users/apple/coding_env/NeuroSAT/tf_neurosat/data/eval/40/'
