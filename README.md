# NeuroSAT

A pytorch implementation of NeuraSAT([github](https://github.com/dselsam/neurosat), [paper](https://arxiv.org/abs/1802.03685))

In this implementation, we use SR(U(10, 40)) for training and SR(40) for testing, achieving the same accuracy 85% as in the original paper. The model was trained on a single K40 gpu for ~3 days following the parameters in the original paper.

## Running Experiments

The repository includes several shell scripts for training and evaluating models on different dataset configurations:

### Training Scripts

- `run_sr3to10.sh`: Trains on SR(3-10) dataset
- `run_sr10to40.sh`: Trains on SR(10-40) dataset
- `run_sr40to100.sh`: Trains on SR(40-100) dataset
- `run_sr200to500.sh`: Trains on SR(200-500) dataset

Each training script performs two steps:
1. Generates the required dataset using `data_maker.py`
2. Runs the training using `src/train.py`

To run a training script:
```bash
./run_sr10to40.sh
```

### Evaluation

To evaluate a trained model on SR(40) dataset, use the `run_eval.sh` script:
```bash
./run_eval.sh
```
This script will first generate an SR(40) dataset for evaluation and then run the evaluation using `src/eval.py`.
