#!/bin/sh

#PBS -N neurosat_sr10t40
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -o $PBS_JOBID.o
#PBS -e $PBS_JOBID.e
#PBS -d /Users/apple/coding_env/NeuroSAT/

# Create directories if they don't exist
mkdir -p data/train
mkdir -p log
mkdir -p data/val

# Step 1: Generate the training dataset using data_maker.py
echo "Generating SR(10-40) dataset for training..."
python src/data_maker.py \
  data/train/sr10-40.pkl \
  log/data_maker_sr10t40.log \
  100000 \
  50000 \
  --min_n 10 \
  --max_n 40 \
  --p_k_2 0.3 \
  --p_geo 0.4

# Step 2: Generate validation dataset if it doesn't exist
if [ ! -f "data/val/sr40.pkl" ]; then
  echo "Generating SR(40) dataset for validation..."
  python src/data_maker.py \
    data/val/sr40.pkl \
    log/data_maker_val_sr40.log \
    1000 \
    50000 \
    --min_n 40 --max_n 40
fi

# Step 3: Run the training script
echo "Starting training on SR(10-40) dataset..."
# Set environment variable to allow unlimited memory for MPS
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python src/train.py \
  --task-name 'neurosat_4th_rnd' \
  --dim 128 \
  --n_rounds 26 \
  --epochs 200 \
  --n_pairs 100000 \
  --max_nodes_per_batch 3000 \
  --data-dir '/Users/apple/coding_env/NeuroSAT/data' \
  --log-dir '/Users/apple/coding_env/NeuroSAT/log' \
  --model-dir '/Users/apple/coding_env/NeuroSAT/model' \
  --val-file 'sr40.pkl'
  # --restore '/Users/apple/coding_env/NeuroSAT/model/neurosat_3rd_rnd_sr10to40_ep200_nr26_d128.pth.tar'
