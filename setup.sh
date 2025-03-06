#!/bin/bash

# Create directories if they don't exist
mkdir -p data/train
mkdir -p data/val
mkdir -p data/eval/40
mkdir -p log
mkdir -p model

# Create a Python virtual environment
echo "Creating Python virtual environment..."
python -m venv neurosat_env
source neurosat_env/bin/activate

# Install dependencies for both implementations
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch tqdm numpy

# Check if running on macOS with Apple Silicon
if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "Detected Apple Silicon Mac, installing MLX and MLX-graphs..."
    pip install mlx mlx-graphs
else
    echo "Not running on Apple Silicon Mac, skipping MLX installation."
    echo "Note: The MLX-graphs implementation will only work on Apple Silicon Macs."
fi

# Generate small test data for verification
echo "Generating small test data..."
python src/data_maker.py data/val/test_small.pkl log/data_maker_test.log 100 1000 --min_n 10 --max_n 20

echo "Setup complete! You can now run the following commands:"
echo ""
echo "# Activate the virtual environment"
echo "source neurosat_env/bin/activate"
echo ""
echo "# Generate training and validation data"
echo "python src/data_maker.py data/train/train_v40_vpb12000_b2604.pkl log/data_maker_train.log 100000 12000 --min_n 10 --max_n 40"
echo "python src/data_maker.py data/val/val_v40_vpb12000_b2604.pkl log/data_maker_val.log 10000 12000 --min_n 10 --max_n 40"
echo ""
echo "# Train the model"
echo "# PyTorch implementation:"
echo "./run_sr10to40.sh"
echo "# MLX-graphs implementation (Apple Silicon only):"
echo "./run_mlx_sr10to40.sh"
echo ""
echo "# Evaluate the model"
echo "# PyTorch implementation:"
echo "./run_eval.sh"
echo "# MLX-graphs implementation (Apple Silicon only):"
echo "./run_eval_mlx.sh"
