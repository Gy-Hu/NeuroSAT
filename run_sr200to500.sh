#!/bin/sh

#PBS -N neurosat_sr200t500
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /Users/apple/coding_env/NeuroSAT/

# Create directories if they don't exist
mkdir -p data/train
mkdir -p log

# Step 1: Generate the dataset using data_maker.py
echo "Generating SR(200-500) dataset for training..."
python src/data_maker.py \
  data/train/sr200-500.pkl \
  log/data_maker_sr200t500.log \
  5000 \
  50000 \
  --min_n 200 \
  --max_n 500 \
  --p_k_2 0.3 \
  --p_geo 0.4

# Step 2: Run the training script
echo "Starting training on SR(200-500) dataset..."
python src/train.py \
  --task-name 'neurosat' \
  --dim 128 \
  --n_rounds 64 \
  --epochs 500 \
  --n_pairs 5000 \
  --max_nodes_per_batch 15000 \
  --data-dir '/Users/apple/coding_env/NeuroSAT/data' \
  --log-dir '/Users/apple/coding_env/NeuroSAT/log' \
  --model-dir '/Users/apple/coding_env/NeuroSAT/model' \
  --val-file 'val_v500_vpb15000_b2564.pkl'
