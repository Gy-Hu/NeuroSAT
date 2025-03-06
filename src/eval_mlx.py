import argparse
import pickle
import os
import time
from tqdm import tqdm

import numpy as np

import mlx.core as mx

from neurosat_mlx import NeuroSATMLX
from data_maker import generate
import mk_problem

from config import parser

def load_model(args, log_file=None):
    net = NeuroSATMLX(args)
    if args.restore:
        if log_file is not None:
            print('restoring from', args.restore, file=log_file, flush=True)
        model = mx.load(args.restore)
        net.update(model['state_dict'])
    
    return net

def predict(net, data):
    net.eval()
    outputs = net.forward(data)
    probs = net.vote
    outputs = mx.sigmoid(outputs)
    preds = mx.where(outputs > 0.5, mx.ones_like(outputs), mx.zeros_like(outputs))
    return preds.tolist(), probs.tolist()

if __name__ == '__main__':
    args = parser.parse_args()
    log_file = open(os.path.join(args.log_dir, args.task_name + '_mlx.log'), 'a+')
    net = load_model(args, log_file)

    TP, TN, FN, FP = 0, 0, 0, 0
    times = []
    for filename in os.listdir(args.data_dir):
        with open(os.path.join(args.data_dir, filename), 'rb') as f:
            xs = pickle.load(f)

        for x in xs:
            start_time = time.time()
            preds, probs = predict(net, x)
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            times.append(duration)

            target = np.array(x.is_sat)
            preds = np.array(preds)
            TP += int(((preds == 1) & (target == 1)).sum())
            TN += int(((preds == 0) & (target == 0)).sum())
            FN += int(((preds == 0) & (target == 1)).sum())
            FP += int(((preds == 1) & (target == 0)).sum())
            
    num_cases = TP + TN + FN + FP
    desc = "%d rnds: tot time %.2f ms for %d cases, avg time: %.2f ms; the pred acc is %.2f, in which TP: %.2f, TN: %.2f, FN: %.2f, FP: %.2f" \
            % (args.n_rounds, sum(times), len(times), sum(times)*1.0/len(times), (TP + TN)*1.0/num_cases, TP*1.0/num_cases, TN*1.0/num_cases, FN*1.0/num_cases, FP*1.0/num_cases)
    print(desc, file=log_file, flush=True)