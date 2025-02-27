#!/bin/sh

#PBS -N neurosat_eval_sr40
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /Users/apple/coding_env/NeuroSAT/

# Create directories if they don't exist
mkdir -p data/eval
mkdir -p log

# Step 1: Generate the evaluation dataset (SR(40)) using data_maker.py
echo "Generating SR(40) dataset for evaluation..."
python src/data_maker.py \
  data/eval/sr40.pkl \
  log/data_maker_eval_sr40.log \
  1000 \
  50000 \
  --min_n 40 \
  --max_n 40 \
  --p_k_2 0.3 \
  --p_geo 0.4

# Step 2: Run the evaluation script
echo "Starting evaluation on SR(40) dataset..."
python src/eval.py \
  --task-name 'neurosat_eval_sr40' \
  --dim 128 \
  --n_rounds 1024 \
  --restore '/Users/apple/coding_env/NeuroSAT/model/neurosat_3rd_rnd_sr10to40_ep200_nr26_d128.pth.tar' \
  --data-dir '/Users/apple/coding_env/NeuroSAT/data/eval'
