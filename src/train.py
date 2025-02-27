import argparse
import pickle
import os
from tqdm import tqdm
import platform

import torch
import torch.nn as nn
import torch.optim as optim

from neurosat import NeuroSAT
from data_maker import generate
import mk_problem

from config import parser

args = parser.parse_args()

# Add MPS (Metal Performance Shaders) support for macOS
if platform.system() == 'Darwin' and torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS (Metal Performance Shaders) for training")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA for training")
else:
    device = torch.device('cpu')
    print("Using CPU for training")

net = NeuroSAT(args, device=device)

task_name = args.task_name + '_sr' + str(args.min_n) + 'to' + str(args.max_n) + '_ep' + str(args.epochs) + '_nr' + str(args.n_rounds) + '_d' + str(args.dim)
log_file = open(os.path.join(args.log_dir, task_name+'.log'), 'a+')
detail_log_file = open(os.path.join(args.log_dir, task_name+'_detail.log'), 'a+')

train, val = None, None
if args.train_file is not None:
  with open(os.path.join(args.data_dir, 'train', args.train_file), 'rb') as f:
    train = pickle.load(f)

with open(os.path.join(args.data_dir, 'val', args.val_file), 'rb') as f:
  val = pickle.load(f)

loss_fn = nn.BCELoss()
optim = optim.Adam(net.parameters(), lr=0.00002, weight_decay=1e-10)
sigmoid  = nn.Sigmoid()

best_acc = 0.0
start_epoch = 0

if train is not None:
  print('num of train batches: ', len(train), file=log_file, flush=True)

print('num of val batches: ', len(val), file=log_file, flush=True)

if args.restore is not None:
  print('restoring from', args.restore, file=log_file, flush=True)
  model = torch.load(args.restore, map_location=device)
  start_epoch = model['epoch']
  best_acc = model['acc']
  net.load_state_dict(model['state_dict'])

for epoch in range(start_epoch, args.epochs):
  if args.train_file is None:
    print('generate data online', file=log_file, flush=True)
    train = generate(args)

  print('==> %d/%d epoch, previous best: %.3f' % (epoch+1, args.epochs, best_acc))
  print('==> %d/%d epoch, previous best: %.3f' % (epoch+1, args.epochs, best_acc), file=log_file, flush=True)
  print('==> %d/%d epoch, previous best: %.3f' % (epoch+1, args.epochs, best_acc), file=detail_log_file, flush=True)
  train_bar = tqdm(train)
  TP, TN, FN, FP = torch.zeros(1, device=device).long(), torch.zeros(1, device=device).long(), torch.zeros(1, device=device).long(), torch.zeros(1, device=device).long()
  net.train()
  for _, prob in enumerate(train_bar):
    optim.zero_grad()
    outputs = net(prob)
    target = torch.tensor(prob.is_sat, device=device).float()
    # print(outputs.shape, target.shape)
    # print(outputs, target)
    outputs = sigmoid(outputs)
    # Make sure both outputs and target are on the same device
    loss = loss_fn(outputs, target.to(outputs.device))
    desc = 'loss: %.4f; ' % (loss.item())

    loss.backward()
    optim.step()

    preds = torch.where(outputs>0.5, torch.ones(outputs.shape, device=device), torch.zeros(outputs.shape, device=device))

    TP += (preds.eq(1) & target.eq(1)).sum()
    TN += (preds.eq(0) & target.eq(0)).sum()
    FN += (preds.eq(0) & target.eq(1)).sum()
    FP += (preds.eq(1) & target.eq(0)).sum()
    TOT = TP + TN + FN + FP
    
    desc += 'acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % ((TP.item()+TN.item())*1.0/TOT.item(), TP.item()*1.0/TOT.item(), TN.item()*1.0/TOT.item(), FN.item()*1.0/TOT.item(), FP.item()*1.0/TOT.item())
    # train_bar.set_description(desc)
    if (_ + 1) % 100 == 0:
      print(desc, file=detail_log_file, flush=True)
  print(desc, file=log_file, flush=True)
  
  val_bar = tqdm(val)
  TP, TN, FN, FP = torch.zeros(1, device=device).long(), torch.zeros(1, device=device).long(), torch.zeros(1, device=device).long(), torch.zeros(1, device=device).long()
  net.eval()
  for _, prob in enumerate(val_bar):
    optim.zero_grad()
    outputs = net(prob)
    target = torch.Tensor(prob.is_sat).to(device).float()
    # print(outputs.shape, target.shape)
    # print(outputs, target)
    outputs = sigmoid(outputs)
    # Make sure both outputs and target are on the same device for loss calculation
    outputs = outputs.to(device)
    target = target.to(device)
    preds = torch.where(outputs>0.5, torch.ones(outputs.shape, device=device), torch.zeros(outputs.shape, device=device))

    TP += (preds.eq(1) & target.eq(1)).sum()
    TN += (preds.eq(0) & target.eq(0)).sum()
    FN += (preds.eq(0) & target.eq(1)).sum()
    FP += (preds.eq(1) & target.eq(0)).sum()
    TOT = TP + TN + FN + FP
    
    desc = 'acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % ((TP.item()+TN.item())*1.0/TOT.item(), TP.item()*1.0/TOT.item(), TN.item()*1.0/TOT.item(), FN.item()*1.0/TOT.item(), FP.item()*1.0/TOT.item())
    # val_bar.set_description(desc)
    if (_ + 1) % 100 == 0:
      print(desc, file=detail_log_file, flush=True)
  print(desc, file=log_file, flush=True)

  acc = (TP.item() + TN.item()) * 1.0 / TOT.item()
  torch.save({'epoch': epoch+1, 'acc': acc, 'state_dict': net.state_dict()}, os.path.join(args.model_dir, task_name+'_last.pth.tar'))
  if acc >= best_acc:
    best_acc = acc
    torch.save({'epoch': epoch+1, 'acc': best_acc, 'state_dict': net.state_dict()}, os.path.join(args.model_dir, task_name+'_best.pth.tar'))
