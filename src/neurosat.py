import torch
import torch.nn as nn
import numpy as np

from mlp import MLP

class NeuroSAT(nn.Module):
  def __init__(self, args, device=None):
    super(NeuroSAT, self).__init__()
    self.args = args
    # Use the provided device or default to CPU
    self.device = device if device is not None else torch.device('cpu')

    self.init_ts = torch.ones(1, device=self.device)
    self.init_ts.requires_grad = False

    self.L_init = nn.Linear(1, args.dim)
    self.C_init = nn.Linear(1, args.dim)

    self.L_msg = MLP(self.args.dim, self.args.dim, self.args.dim)
    self.C_msg = MLP(self.args.dim, self.args.dim, self.args.dim)

    self.L_update = nn.LSTM(self.args.dim*2, self.args.dim)
    # self.L_norm   = nn.LayerNorm(self.args.dim)
    self.C_update = nn.LSTM(self.args.dim, self.args.dim)
    # self.C_norm   = nn.LayerNorm(self.args.dim)

    self.L_vote = MLP(self.args.dim, self.args.dim, 1)

    self.denom = torch.sqrt(torch.tensor([self.args.dim], device=self.device))
    
    # Move all model parameters to the specified device
    self.to(self.device)
    
  # Add helper method to move problem data to device
  def _move_problem_to_device(self, problem):
    # This is a simple object with attributes, not a proper PyTorch structure with a to() method
    # So we manually move its tensor components
    # We preserve the object itself and just change its tensor attributes
    if hasattr(problem, "L_unpack_indices"):
      problem.L_unpack_indices = [idx for idx in problem.L_unpack_indices]  # Keep this as a list
    return problem

  def forward(self, problem):
    # Move problem data to the correct device
    problem = self._move_problem_to_device(problem)
    
    n_vars    = problem.n_vars
    n_lits    = problem.n_lits
    n_clauses = problem.n_clauses
    n_probs   = len(problem.is_sat)
    # print(n_vars, n_lits, n_clauses, n_probs)

    # Ensure all tensors are moved to the correct device
    # Convert list of arrays to single numpy array for better performance
    ts_L_unpack_indices = torch.tensor(np.array(problem.L_unpack_indices), device=self.device).t().long()
    
    init_ts = self.init_ts
    # 1 x n_lits x dim & 1 x n_clauses x dim
    L_init = self.L_init(init_ts).view(1, 1, -1)
    # print(L_init.shape)
    L_init = L_init.repeat(1, n_lits, 1)
    C_init = self.C_init(init_ts).view(1, 1, -1)
    # print(C_init.shape)
    C_init = C_init.repeat(1, n_clauses, 1)

    # print(L_init.shape, C_init.shape)

    # Make sure all LSTM states are on the correct device
    h0_L = L_init.to(self.device)
    c0_L = torch.zeros(1, n_lits, self.args.dim, device=self.device)
    L_state = (h0_L, c0_L)
    
    h0_C = C_init.to(self.device)
    c0_C = torch.zeros(1, n_clauses, self.args.dim, device=self.device)
    C_state = (h0_C, c0_C)
    # MPS doesn't support sparse tensors, so we need to handle this differently
    if self.device.type == 'mps':
        # Create a dense tensor directly without going through sparse format
        L_unpack = torch.zeros(n_lits, n_clauses, device=self.device)
        for i in range(len(problem.L_unpack_indices[0])):
            L_unpack[ts_L_unpack_indices[0, i], ts_L_unpack_indices[1, i]] = 1.0
    else:
        L_unpack = torch.sparse.FloatTensor(
            ts_L_unpack_indices,
            torch.ones(problem.n_cells, device=self.device),
            torch.Size([n_lits, n_clauses])
        ).to_dense().to(self.device)

    # print(ts_L_unpack_indices.shape)

    for _ in range(self.args.n_rounds):
      # n_lits x dim
      L_hidden = L_state[0].squeeze(0).to(self.device)
      L_pre_msg = self.L_msg(L_hidden)
      # (n_clauses x n_lits) x (n_lits x dim) = n_clauses x dim
      LC_msg = torch.matmul(L_unpack.t().to(self.device), L_pre_msg.to(self.device))
      # print(L_hidden.shape, L_pre_msg.shape, LC_msg.shape)

      # Make sure LSTM input is on the correct device
      LC_msg = LC_msg.to(self.device)
      _, C_state = self.C_update(LC_msg.unsqueeze(0), C_state)
      # print('C_state',C_state[0].shape, C_state[1].shape)

      # n_clauses x dim
      C_hidden = C_state[0].squeeze(0).to(self.device)
      C_pre_msg = self.C_msg(C_hidden)
      # (n_lits x n_clauses) x (n_clauses x dim) = n_lits x dim
      CL_msg = torch.matmul(L_unpack.to(self.device), C_pre_msg.to(self.device))
      # print(C_hidden.shape, C_pre_msg.shape, CL_msg.shape)

      # Create the concatenated tensor for LSTM input
      flipped = self.flip(L_state[0].squeeze(0), n_vars)
      cat_tensor = torch.cat([CL_msg.to(self.device), flipped.to(self.device)], dim=1).to(self.device)
      _, L_state = self.L_update(cat_tensor.unsqueeze(0), L_state)
      # print('L_state',C_state[0].shape, C_state[1].shape)
      
    # Ensure final tensors are on correct device
    logits = L_state[0].squeeze(0).to(self.device)
    clauses = C_state[0].squeeze(0).to(self.device)

    # print(logits.shape, clauses.shape)
    vote = self.L_vote(logits)
    # print('vote', vote.shape)
    vote_join = torch.cat([vote[:n_vars, :], vote[n_vars:, :]], dim=1).to(self.device)
    # print('vote_join', vote_join.shape)
    self.vote = vote_join
    vote_join = vote_join.view(n_probs, -1, 2).view(n_probs, -1)
    vote_mean = torch.mean(vote_join, dim=1)
    # print('mean', vote_mean.shape)
    return vote_mean.to(self.device)

  def flip(self, msg, n_vars):
    # Ensure tensors are on the correct device
    msg = msg.to(self.device)
    return torch.cat([msg[n_vars:2*n_vars, :], msg[:n_vars, :]], dim=0).to(self.device)
