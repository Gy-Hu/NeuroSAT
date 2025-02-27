# NeuroSAT

A pytorch implementation of NeuraSAT([github](https://github.com/dselsam/neurosat), [paper](https://arxiv.org/abs/1802.03685))

In this implementation, we use SR(U(10, 40)) for training and SR(40) for testing, achieving the same accuracy 85% as in the original paper. The model was trained on a single K40 gpu for ~3 days following the parameters in the original paper.

## Data Preparation

Before running any experiment scripts, you need to generate the SAT problem instances using the `src/data_maker.py` script:

```bash
python src/data_maker.py <out_dir> <gen_log> <n_pairs> <max_nodes_per_batch> [other options]
```

For example, to generate the SR(40) dataset for testing, you could run something like:
```bash
python src/data_maker.py data/val/sr40.pkl data/val/sr40.log 1000 50000 --min_n 40 --max_n 40
```

To generate the SR(U(10, 40)) dataset for training:
```bash
python src/data_maker.py data/train/sr10-40.pkl data/train/sr10-40.log 10000 50000 --min_n 10 --max_n 40
```

### Key Parameters:
- `<out_dir>`: Output path for the generated pickle file containing the SAT problems
- `<gen_log>`: Path for the generation log file
- `<n_pairs>`: Number of SAT problem pairs to generate
- `<max_nodes_per_batch>`: Maximum number of nodes per batch
- `--min_n` and `--max_n`: Range of variables in the SAT problems (SR(40) means min_n=max_n=40)
- `--p_k_2` and `--p_geo`: Distribution parameters for clause generation (defaults: 0.3 and 0.4)

You'll also need to generate validation datasets for training scripts. For example:
```bash
# Generate validation dataset for SR(40)
python src/data_maker.py data/val/sr40.pkl data/val/sr40.log 1000 50000 --min_n 40 --max_n 40
```

## Running Experiments

The repository includes several shell scripts for training and evaluating models on different dataset configurations:

### Training Scripts

- `run_sr3to10.sh`: Trains on SR(3-10) dataset
- `run_sr10to40.sh`: Trains on SR(10-40) dataset
- `run_sr40to100.sh`: Trains on SR(40-100) dataset
- `run_sr200to500.sh`: Trains on SR(200-500) dataset

Each training script performs three steps:
1. Generates the required training dataset using `data_maker.py`
2. Generates a validation dataset if it doesn't exist yet
3. Runs the training using `src/train.py`

The scripts have been configured to automatically create all necessary directories and datasets.

To run a training script (example):
```bash
./run_sr10to40.sh
```

### Evaluation

To evaluate a trained model on SR(40) dataset, use the `run_eval.sh` script:
```bash
./run_eval.sh
```
This script will first generate an SR(40) dataset specifically for evaluation and then run the evaluation using `src/eval.py`.
