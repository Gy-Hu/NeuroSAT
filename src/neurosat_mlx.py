import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.nn.linear import Linear
from mlx_graphs.utils import scatter


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.l1 = Linear(in_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, hidden_dim)
        self.l3 = Linear(hidden_dim, out_dim)

    def __call__(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


class LiteralToClauseMessage(MessagePassing):
    def __init__(self, dim, **kwargs):
        super(LiteralToClauseMessage, self).__init__(aggr="add", **kwargs)
        self.msg_mlp = MLP(dim, dim, dim)
        
    def __call__(self, edge_index, node_features, **kwargs):
        # Process literal features through MLP before message passing
        literal_features = self.msg_mlp(node_features)
        
        # Propagate messages from literals to clauses
        return self.propagate(
            edge_index=edge_index,
            node_features=literal_features,
        )


class ClauseToLiteralMessage(MessagePassing):
    def __init__(self, dim, **kwargs):
        super(ClauseToLiteralMessage, self).__init__(aggr="add", **kwargs)
        self.msg_mlp = MLP(dim, dim, dim)
        
    def __call__(self, edge_index, node_features, **kwargs):
        # Process clause features through MLP before message passing
        clause_features = self.msg_mlp(node_features)
        
        # Propagate messages from clauses to literals
        return self.propagate(
            edge_index=edge_index,
            node_features=clause_features,
        )


class NeuroSATMLX(nn.Module):
    def __init__(self, args):
        super(NeuroSATMLX, self).__init__()
        self.args = args
        
        # Initialize embeddings for literals and clauses
        self.L_init = Linear(1, args.dim)
        self.C_init = Linear(1, args.dim)
        
        # Message passing networks
        self.L_to_C = LiteralToClauseMessage(args.dim)
        self.C_to_L = ClauseToLiteralMessage(args.dim)
        
        # Update networks
        self.L_update = nn.LSTM(args.dim * 2, args.dim)
        self.C_update = nn.LSTM(args.dim, args.dim)
        
        # Final voting layer
        self.L_vote = MLP(args.dim, args.dim, 1)
        
        # Normalization factor
        self.denom = mx.sqrt(mx.array([args.dim], dtype=mx.float32))
        
    def flip(self, msg, n_vars):
        """Flip the literals (negate them)"""
        return mx.concatenate([msg[n_vars:2*n_vars, :], msg[:n_vars, :]], axis=0)
    
    def forward(self, problem):
        n_vars = problem.n_vars
        n_lits = problem.n_lits
        n_clauses = problem.n_clauses
        n_probs = len(problem.is_sat)
        
        #print(f"Problem dimensions: n_vars={n_vars}, n_lits={n_lits}, n_clauses={n_clauses}, n_probs={n_probs}")
        
        # Get L_unpack_indices and transpose to match PyTorch's format
        L_unpack_indices = mx.array(problem.L_unpack_indices)
        #print(f"L_unpack_indices shape: {L_unpack_indices.shape}")
        
        # Initialize literal and clause embeddings
        init_ts = mx.ones((1,))
        L_init = self.L_init(init_ts).reshape(1, 1, -1)
        L_init = mx.broadcast_to(L_init, (1, n_lits, self.args.dim))
        C_init = self.C_init(init_ts).reshape(1, 1, -1)
        C_init = mx.broadcast_to(C_init, (1, n_clauses, self.args.dim))
        
        #print(f"L_init shape: {L_init.shape}, C_init shape: {C_init.shape}")
        
        # Initialize LSTM states
        L_state = (L_init, mx.zeros((1, n_lits, self.args.dim)))
        C_state = (C_init, mx.zeros((1, n_clauses, self.args.dim)))
        
        # Create a dense adjacency matrix for message passing (similar to PyTorch's to_dense())
        import numpy as np
        L_unpack_np = np.zeros((n_lits, n_clauses))
        indices_np = problem.L_unpack_indices
        for i in range(len(indices_np)):
            src, dst = int(indices_np[i][0]), int(indices_np[i][1])
            L_unpack_np[src, dst] = 1.0
        
        # Convert to MLX array
        L_unpack = mx.array(L_unpack_np)
        #print(f"L_unpack shape: {L_unpack.shape}")
        
        # Message passing rounds
        for round_idx in range(self.args.n_rounds):
            #print(f"\nRound {round_idx+1}/{self.args.n_rounds}")
            
            # Literal to Clause messages
            L_hidden = L_state[0].squeeze(0)
            #print(f"L_hidden shape: {L_hidden.shape}")
            
            # Process literal features through MLP (equivalent to L_msg in PyTorch)
            L_pre_msg = self.L_to_C.msg_mlp(L_hidden)
            #print(f"L_pre_msg shape: {L_pre_msg.shape}")
            
            # Perform message passing (equivalent to torch.matmul(L_unpack.t(), L_pre_msg))
            LC_msg = mx.matmul(L_unpack.transpose(), L_pre_msg)
            #print(f"LC_msg shape: {LC_msg.shape}")
            
            # Update clause states (equivalent to _, C_state = self.C_update(LC_msg.unsqueeze(0), C_state))
            LC_msg_reshaped = LC_msg.reshape(1, n_clauses, -1)
            #print(f"LC_msg_reshaped shape: {LC_msg_reshaped.shape}")
            #print(f"C_state[0] shape: {C_state[0].shape}, C_state[1] shape: {C_state[1].shape}")
            
            # Get the hidden and cell states from the LSTM
            hidden, cell = self.C_update(LC_msg_reshaped, C_state[0], C_state[1])
            #print(f"After C_update - hidden shape: {hidden.shape}, cell shape: {cell.shape}")
            
            # Take only the last time step (MLX LSTM returns all time steps)
            # If hidden has shape (1, n_clauses, n_clauses, dim), we need to reshape it
            if len(hidden.shape) == 4:
                # Take the last time step
                hidden = hidden[:, -1:, :, :]
                cell = cell[:, -1:, :, :]
                # Reshape to (1, n_clauses, dim)
                hidden = hidden.reshape(1, n_clauses, -1)
                cell = cell.reshape(1, n_clauses, -1)
            
            #print(f"After reshaping - hidden shape: {hidden.shape}, cell shape: {cell.shape}")
            C_state = (hidden, cell)
            
            # Clause to Literal messages
            C_hidden = C_state[0].squeeze(0)
            #print(f"C_hidden shape: {C_hidden.shape}")
            
            # Process clause features through MLP (equivalent to C_msg in PyTorch)
            C_pre_msg = self.C_to_L.msg_mlp(C_hidden)
            #print(f"C_pre_msg shape: {C_pre_msg.shape}")
            
            # Perform message passing (equivalent to torch.matmul(L_unpack, C_pre_msg))
            CL_msg = mx.matmul(L_unpack, C_pre_msg)
            #print(f"CL_msg shape: {CL_msg.shape}")
            
            # Flip the literals
            flipped = self.flip(L_state[0].squeeze(0), n_vars)
            #print(f"flipped shape: {flipped.shape}")
            
            # Concatenate messages and flipped literals
            try:
                concat_input = mx.concatenate([CL_msg, flipped], axis=1).reshape(1, n_lits, -1)
                #print(f"concat_input shape: {concat_input.shape}")
            except Exception as e:
                #print(f"Error in concatenation: {e}")
                #print(f"CL_msg shape: {CL_msg.shape}, flipped shape: {flipped.shape}")
                # Try to fix the shapes
                if len(CL_msg.shape) != len(flipped.shape):
                    if len(CL_msg.shape) == 3:
                        CL_msg = CL_msg[0]  # Take first slice if 3D
                    elif len(flipped.shape) == 3:
                        flipped = flipped[0]  # Take first slice if 3D
                #print(f"After fixing - CL_msg shape: {CL_msg.shape}, flipped shape: {flipped.shape}")
                concat_input = mx.concatenate([CL_msg, flipped], axis=1).reshape(1, n_lits, -1)
            
            # Update literal states (equivalent to _, L_state = self.L_update(concat_input, L_state))
            #print(f"L_state[0] shape: {L_state[0].shape}, L_state[1] shape: {L_state[1].shape}")
            hidden, cell = self.L_update(concat_input, L_state[0], L_state[1])
            #print(f"After L_update - hidden shape: {hidden.shape}, cell shape: {cell.shape}")
            
            # Take only the last time step (MLX LSTM returns all time steps)
            # If hidden has shape (1, n_lits, n_lits, dim), we need to reshape it
            if len(hidden.shape) == 4:
                # Take the last time step
                hidden = hidden[:, -1:, :, :]
                cell = cell[:, -1:, :, :]
                # Reshape to (1, n_lits, dim)
                hidden = hidden.reshape(1, n_lits, -1)
                cell = cell.reshape(1, n_lits, -1)
            
            #print(f"After reshaping L_state - hidden shape: {hidden.shape}, cell shape: {cell.shape}")
            L_state = (hidden, cell)
        
        # Extract final states
        logits = L_state[0].squeeze(0)
        
        # Voting layer
        vote = self.L_vote(logits)
        vote_join = mx.concatenate([vote[:n_vars, :], vote[n_vars:, :]], axis=1)
        self.vote = vote_join
        vote_join = vote_join.reshape(n_probs, -1, 2).reshape(n_probs, -1)
        vote_mean = mx.mean(vote_join, axis=1)
        
        return vote_mean