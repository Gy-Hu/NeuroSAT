#!/bin/sh

#PBS -N neurosat_sr40t100
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /Users/apple/coding_env/NeuroSAT/

# Create directories if they don't exist
mkdir -p data/train
mkdir -p log

# Step 1: Generate the dataset using data_maker.py
echo "Generating SR(40-100) dataset for training..."
python src/data_maker.py \
  data/train/sr40-100.pkl \
  log/data_maker_sr40t100.log \
  10000 \
  50000 \
  --min_n 40 \
  --max_n 100 \
  --p_k_2 0.3 \
  --p_geo 0.4

# Step 2: Run the training script
echo "Starting training on SR(40-100) dataset..."
python src/train.py \
  --task-name 'neurosat_3rd_rnd' \
  --dim 128 \
  --n_rounds 32 \
  --epochs 200 \
  --n_pairs 10000 \
  --max_nodes_per_batch 12000 \
  --data-dir '/Users/apple/coding_env/NeuroSAT/data' \
  --log-dir '/Users/apple/coding_env/NeuroSAT/log' \
  --model-dir '/Users/apple/coding_env/NeuroSAT/model' \
  --val-file 'val_v100_vpb12000_b1284.pkl' \
  --restore '/Users/apple/coding_env/NeuroSAT/model/neurosat_2nd_rnd_sr40to100_ep200_nr32_d128_last.pth.tar'
