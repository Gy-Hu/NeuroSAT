#!/bin/sh

#PBS -N neurosat_sr3t10
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /Users/apple/coding_env/NeuroSAT/

# Create directories if they don't exist
mkdir -p data/train
mkdir -p log
mkdir -p data/val

# Step 1: Generate the training dataset using data_maker.py
echo "Generating SR(3-10) dataset for training..."
python src/data_maker.py \
  data/train/sr3-10.pkl \
  log/data_maker_sr3t10.log \
  100000 \
  50000 \
  --min_n 3 \
  --max_n 10 \
  --p_k_2 0.3 \
  --p_geo 0.4

# Step 2: Generate validation dataset if it doesn't exist
if [ ! -f "data/val/sr10.pkl" ]; then
  echo "Generating SR(10) dataset for validation..."
  python src/data_maker.py \
    data/val/sr10.pkl \
    log/data_maker_val_sr10.log \
    1000 \
    50000 \
    --min_n 10 --max_n 10
fi

# Step 3: Run the training script
echo "Starting training on SR(3-10) dataset..."
python src/train.py \
  --task-name 'neurosat_No2' \
  --dim 128 \
  --n_rounds 26 \
  --epochs 10 \
  --n_pairs 100000 \
  --max_nodes_per_batch 12000 \
  --data-dir '/Users/apple/coding_env/NeuroSAT/data' \
  --log-dir '/Users/apple/coding_env/NeuroSAT/log' \
  --model-dir '/Users/apple/coding_env/NeuroSAT/model' \
  --val-file 'sr10.pkl'
