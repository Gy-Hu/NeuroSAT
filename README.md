# NeuroSAT

A PyTorch and MLX-graphs implementation of NeuroSAT ([github](https://github.com/dselsam/neurosat), [paper](https://arxiv.org/abs/1802.03685))

## Overview

In this implementation, we use SR(U(10, 40)) for training and SR(40) for testing, achieving the same accuracy 85% as in the original paper. The original PyTorch model was trained on a single K40 GPU for ~3 days following the parameters in the original paper.

We've also added a new implementation using MLX-graphs, which is optimized for Apple Silicon hardware and provides better performance on Mac devices.

## Requirements

### PyTorch Implementation
- Python 3.6+
- PyTorch 1.0+
- tqdm
- numpy

### MLX-graphs Implementation
- Python 3.10+
- MLX 0.18+
- MLX-graphs
- tqdm
- numpy

## Installation

### Quick Setup

The easiest way to get started is to use the provided setup script:

```bash
# Clone the repository
git clone https://github.com/yourusername/neurosat.git
cd neurosat

# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

This script will:
- Create necessary directories
- Set up a Python virtual environment
- Install dependencies for both implementations
- Generate a small test dataset

### Manual Setup

If you prefer to set up manually:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neurosat.git
cd neurosat
```

2. Create necessary directories:
```bash
mkdir -p data/train data/val data/eval/40 log model
```

3. Install dependencies:
```bash
# For PyTorch implementation
pip install torch tqdm numpy

# For MLX-graphs implementation (on Apple Silicon Macs)
pip install mlx mlx-graphs tqdm numpy
```

## Data Generation

To generate training and validation data:

```bash
# Generate training data
python src/data_maker.py data/train/train_v40_vpb12000_b2604.pkl log/data_maker_train.log 100000 12000 --min_n 10 --max_n 40

# Generate validation data
python src/data_maker.py data/val/val_v40_vpb12000_b2604.pkl log/data_maker_val.log 10000 12000 --min_n 10 --max_n 40
```

## Training

### PyTorch Implementation

To train the model using PyTorch:

```bash
./run_sr10to40.sh
```

Or manually:

```bash
python src/train.py \
  --task-name 'neurosat' \
  --dim 128 \
  --n_rounds 26 \
  --epochs 200 \
  --n_pairs 100000 \
  --max_nodes_per_batch 12000 \
  --gen_log 'log/data_maker_sr10t40.log' \
  --min_n 10 \
  --max_n 40 \
  --val-file 'val_v40_vpb12000_b2604.pkl'
```

### MLX-graphs Implementation

To train the model using MLX-graphs (optimized for Apple Silicon):

```bash
./run_mlx_sr10to40.sh
```

Or manually:

```bash
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
```

## Evaluation

### PyTorch Implementation

To evaluate the trained PyTorch model:

```bash
./run_eval.sh
```

Or manually:

```bash
python src/eval.py \
  --task-name 'neurosat_eval_sr40' \
  --dim 128 \
  --n_rounds 1024 \
  --restore 'model/neurosat_3rd_rnd_sr10to40_ep200_nr26_d128.pth.tar' \
  --data-dir 'data/eval/40/'
```

### MLX-graphs Implementation

To evaluate the trained MLX-graphs model:

```bash
./run_eval_mlx.sh
```

Or manually:

```bash
python src/eval_mlx.py \
  --task-name 'neurosat_mlx_eval_sr40' \
  --dim 128 \
  --n_rounds 1024 \
  --restore 'model/neurosat_mlx_sr10to40_ep200_nr26_d128_best.npz' \
  --data-dir 'data/eval/40/'
```

## Implementation Details

### PyTorch Implementation
The original implementation uses PyTorch for graph learning with message passing between literals and clauses.

### MLX-graphs Implementation
The MLX-graphs implementation leverages the MessagePassing class from MLX-graphs to implement the message passing mechanism more efficiently on Apple Silicon hardware. This implementation benefits from:

- Better performance on Apple Silicon hardware
- Unified memory architecture allowing for larger graphs
- More efficient message passing operations
- Native support for graph operations through the MLX-graphs library

## Results

Both implementations achieve approximately 85% accuracy on the SR(40) test set, matching the results reported in the original paper.
