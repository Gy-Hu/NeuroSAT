#!/bin/sh

#PBS -N neurosat_sr200t500
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /Users/apple/coding_env/NeuroSAT/

# Create directories if they don't exist
mkdir -p data/train
mkdir -p log
mkdir -p data/val

# Step 1: Generate the training dataset using data_maker.py
echo "Generating SR(200-500) dataset for training..."
python src/data_maker.py \
  data/train/sr200-500.pkl \
  log/data_maker_sr200t500.log \
  50000 \
  50000 \
  --min_n 200 \
  --max_n 500 \
  --p_k_2 0.3 \
  --p_geo 0.4

# Step 2: Generate validation dataset if it doesn't exist
if [ ! -f "data/val/sr500.pkl" ]; then
  echo "Generating SR(500) dataset for validation..."
  python src/data_maker.py \
    data/val/sr500.pkl \
    log/data_maker_val_sr500.log \
    1000 \
    50000 \
    --min_n 500 --max_n 500
fi

# Step 3: Run the training script
echo "Starting training on SR(200-500) dataset..."
python src/train.py \
  --task-name 'neurosat' \
  --dim 128 \
  --n_rounds 64 \
  --epochs 500 \
  --n_pairs 50000 \
  --auto_batch_size \
  --small_batch_size 15000 \
  --medium_batch_size 8000 \
  --large_batch_size 3000 \
  --data-dir '/Users/apple/coding_env/NeuroSAT/data' \
  --log-dir '/Users/apple/coding_env/NeuroSAT/log' \
  --model-dir '/Users/apple/coding_env/NeuroSAT/model' \
  --val-file 'sr500.pkl'
