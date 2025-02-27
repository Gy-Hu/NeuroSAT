import torch
import torch.nn as nn
import numpy as np
import gc

from mlp import MLP

class NeuroSAT(nn.Module):
  def __init__(self, args, device=None):
    super(NeuroSAT, self).__init__()
    self.args = args
    # Use the provided device or default to CPU
    self.device = device if device is not None else torch.device('cpu')
    # Store whether garbage collection is disabled
    self.disable_gc = args.disable_gc if hasattr(args, 'disable_gc') else False

    # Create tensors directly on the target device to avoid transfers
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
    
  # Helper method to conditionally collect garbage
  def _maybe_collect_garbage(self):
    if not self.disable_gc:
      if self.device.type == 'mps':
        if hasattr(torch.mps, 'empty_cache'):
          torch.mps.empty_cache()
      elif self.device.type == 'cuda':
        torch.cuda.empty_cache()
      gc.collect()

  def forward(self, problem):
    # Move problem data to the correct device
    problem = self._move_problem_to_device(problem)
    
    n_vars    = problem.n_vars
    n_lits    = problem.n_lits
    n_clauses = problem.n_clauses
    n_probs   = len(problem.is_sat)
    # print(n_vars, n_lits, n_clauses, n_probs)

    # Create tensors directly on the target device to avoid transfers
    # Convert list of arrays to single numpy array for better performance
    ts_L_unpack_indices = torch.tensor(np.array(problem.L_unpack_indices), device=self.device).t().long()
    
    # Create all tensors directly on the device to avoid transfers
    init_ts = self.init_ts
    # 1 x n_lits x dim & 1 x n_clauses x dim
    L_init = self.L_init(init_ts).view(1, 1, -1)
    L_init = L_init.repeat(1, n_lits, 1)
    C_init = self.C_init(init_ts).view(1, 1, -1)
    
    # Free memory after initialization
    init_ts = None
    self._maybe_collect_garbage()
    
    C_init = C_init.repeat(1, n_clauses, 1)

    # Create LSTM states directly on the device
    h0_L = L_init  # Already on device, no need for .to(self.device)
    c0_L = torch.zeros(1, n_lits, self.args.dim, device=self.device)
    L_state = (h0_L, c0_L)
    
    h0_C = C_init  # Already on device, no need for .to(self.device)
    c0_C = torch.zeros(1, n_clauses, self.args.dim, device=self.device)
    C_state = (h0_C, c0_C)
    
    # MPS doesn't support sparse tensors, so we need to handle this differently
    if self.device.type == 'mps':
        # Create a dense tensor directly without going through sparse format
        L_unpack = torch.zeros(n_lits, n_clauses, dtype=torch.float32, device=self.device)
        for i in range(len(problem.L_unpack_indices[0])):
            L_unpack[ts_L_unpack_indices[0, i], ts_L_unpack_indices[1, i]] = 1.0
        # Free memory after creating L_unpack
        ts_L_unpack_indices = None
        self._maybe_collect_garbage()
    else:
        L_unpack = torch.sparse.FloatTensor(
            ts_L_unpack_indices,
            torch.ones(problem.n_cells, device=self.device),
            torch.Size([n_lits, n_clauses])
        ).to_dense()  # Already on device, no need for .to(self.device)

    for _ in range(self.args.n_rounds):
      # Only clear cache if garbage collection is enabled
      if not self.disable_gc:
        if self.device.type == 'mps':
          if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
          torch.cuda.empty_cache()
        
      # Get hidden state - already on device, no need for .to(self.device)
      L_hidden = L_state[0].squeeze(0)
      L_pre_msg = self.L_msg(L_hidden)
      
      # Use more memory-efficient matrix multiplication
      L_unpack_t = L_unpack.t()
      LC_msg = torch.matmul(L_unpack_t, L_pre_msg)
      
      # Clean up intermediate tensors
      L_unpack_t = None
      L_pre_msg = None
      self._maybe_collect_garbage()

      # LSTM update - all tensors already on device
      _, C_state = self.C_update(LC_msg.unsqueeze(0), C_state)
      
      LC_msg = None
      self._maybe_collect_garbage()

      # Get hidden state - already on device, no need for .to(self.device)
      C_hidden = C_state[0].squeeze(0)
      C_pre_msg = self.C_msg(C_hidden)
      
      # Matrix multiplication - all tensors already on device
      CL_msg = torch.matmul(L_unpack, C_pre_msg)
      
      C_pre_msg = None
      self._maybe_collect_garbage()

      # Create the concatenated tensor for LSTM input
      flipped = self.flip(L_state[0].squeeze(0), n_vars)
      cat_tensor = torch.cat([CL_msg, flipped], dim=1)
      
      CL_msg = None
      flipped = None
      
      # LSTM update - all tensors already on device
      _, L_state = self.L_update(cat_tensor.unsqueeze(0), L_state)
      
      cat_tensor = None
      self._maybe_collect_garbage()
      
    # Get final tensors - already on device, no need for .to(self.device)
    logits = L_state[0].squeeze(0)
    
    # Free memory before final computation
    L_state = None
    C_state = None
    L_unpack = None
    self._maybe_collect_garbage()

    vote = self.L_vote(logits)
    vote_join = torch.cat([vote[:n_vars, :], vote[n_vars:, :]], dim=1)
    
    vote = None
    logits = None
    self._maybe_collect_garbage()
    
    self.vote = vote_join
    vote_join = vote_join.view(n_probs, -1, 2).view(n_probs, -1)
    vote_mean = torch.mean(vote_join, dim=1)
    
    # Already on device, no need for final .to(self.device)
    return vote_mean

  def flip(self, msg, n_vars):
    # No need to move to device, msg is already on the correct device
    return torch.cat([msg[n_vars:2*n_vars, :], msg[:n_vars, :]], dim=0)
    
  def __del__(self):
    # Clean up memory when the model is deleted
    if not self.disable_gc:
      if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
      gc.collect()
