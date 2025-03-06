import argparse
import pickle
import os
from tqdm import tqdm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from neurosat_mlx import NeuroSATMLX
from data_maker import generate
import mk_problem

from config import parser

args = parser.parse_args()

# Create the NeuroSAT model using MLX
net = NeuroSATMLX(args)

task_name = args.task_name + '_sr' + str(args.min_n) + 'to' + str(args.max_n) + '_ep' + str(args.epochs) + '_nr' + str(args.n_rounds) + '_d' + str(args.dim)
log_file = open(os.path.join(args.log_dir, task_name+'.log'), 'a+')
detail_log_file = open(os.path.join(args.log_dir, task_name+'_detail.log'), 'a+')

train, val = None, None
if args.train_file is not None:
    with open(os.path.join(args.data_dir, 'train', args.train_file), 'rb') as f:
        train = pickle.load(f)

with open(os.path.join(args.data_dir, 'val', args.val_file), 'rb') as f:
    val = pickle.load(f)

# Define loss function and optimizer
loss_fn = nn.losses.binary_cross_entropy
optimizer = optim.Adam(learning_rate=0.00002)

best_acc = 0.0
start_epoch = 0

if train is not None:
    print('num of train batches: ', len(train), file=log_file, flush=True)

print('num of val batches: ', len(val), file=log_file, flush=True)

if args.restore is not None:
    print('restoring from', args.restore, file=log_file, flush=True)
    model = mx.load(args.restore)
    start_epoch = model['epoch']
    best_acc = model['acc']
    net.update(model['state_dict'])

# Define forward function for computing loss and gradients
def forward_fn(model, prob):
    outputs = model.forward(prob)
    target = mx.array(prob.is_sat, dtype=mx.float32)
    outputs = mx.sigmoid(outputs)
    loss = mx.mean(loss_fn(outputs, target))
    return loss, outputs

# Get value and gradient function
loss_and_grad_fn = nn.value_and_grad(net, forward_fn)

for epoch in range(start_epoch, args.epochs):
    if args.train_file is None:
        print('generate data online', file=log_file, flush=True)
        train = generate(args)

    print('==> %d/%d epoch, previous best: %.3f' % (epoch+1, args.epochs, best_acc))
    print('==> %d/%d epoch, previous best: %.3f' % (epoch+1, args.epochs, best_acc), file=log_file, flush=True)
    print('==> %d/%d epoch, previous best: %.3f' % (epoch+1, args.epochs, best_acc), file=detail_log_file, flush=True)
    
    train_bar = tqdm(train)
    TP, TN, FN, FP = 0, 0, 0, 0
    net.train()
    
    for _, prob in enumerate(train_bar):
        # Forward pass and compute gradients
        (loss, outputs), grads = loss_and_grad_fn(net, prob)
        
        # Update parameters
        optimizer.update(net, grads)
        
        # Compute metrics
        target = mx.array(prob.is_sat, dtype=mx.float32)
        preds = mx.where(outputs > 0.5, mx.ones_like(outputs), mx.zeros_like(outputs))
        
        TP += mx.sum(mx.logical_and(preds == 1, target == 1)).item()
        TN += mx.sum(mx.logical_and(preds == 0, target == 0)).item()
        FN += mx.sum(mx.logical_and(preds == 0, target == 1)).item()
        FP += mx.sum(mx.logical_and(preds == 1, target == 0)).item()
        TOT = TP + TN + FN + FP
        
        desc = 'loss: %.4f; ' % (loss.item())
        desc += 'acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % ((TP+TN)*1.0/TOT, TP*1.0/TOT, TN*1.0/TOT, FN*1.0/TOT, FP*1.0/TOT)
        
        if (_ + 1) % 100 == 0:
            print(desc, file=detail_log_file, flush=True)
    
    print(desc, file=log_file, flush=True)
    
    # Validation
    val_bar = tqdm(val)
    TP, TN, FN, FP = 0, 0, 0, 0
    net.eval()
    
    for _, prob in enumerate(val_bar):
        # Forward pass
        outputs = net.forward(prob)
        target = mx.array(prob.is_sat, dtype=mx.float32)
        outputs = mx.sigmoid(outputs)
        preds = mx.where(outputs > 0.5, mx.ones_like(outputs), mx.zeros_like(outputs))
        
        TP += mx.sum(mx.logical_and(preds == 1, target == 1)).item()
        TN += mx.sum(mx.logical_and(preds == 0, target == 0)).item()
        FN += mx.sum(mx.logical_and(preds == 0, target == 1)).item()
        FP += mx.sum(mx.logical_and(preds == 1, target == 0)).item()
        TOT = TP + TN + FN + FP
        
        desc = 'acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % ((TP+TN)*1.0/TOT, TP*1.0/TOT, TN*1.0/TOT, FN*1.0/TOT, FP*1.0/TOT)
        
        if (_ + 1) % 100 == 0:
            print(desc, file=detail_log_file, flush=True)
    
    print(desc, file=log_file, flush=True)
    
    # Save model
    acc = (TP + TN) * 1.0 / TOT
    mx.save(os.path.join(args.model_dir, task_name+'_last.npz'), 
            {'epoch': epoch+1, 'acc': acc, 'state_dict': net.parameters()})
    
    if acc >= best_acc:
        best_acc = acc
        mx.save(os.path.join(args.model_dir, task_name+'_best.npz'), 
                {'epoch': epoch+1, 'acc': best_acc, 'state_dict': net.parameters()})