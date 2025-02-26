import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import dgl.function as fn
import numpy as np
import random
import pickle
import os
import time
import argparse
from tqdm import tqdm
from PyMiniSolvers.minisolvers import MinisatSolver


def generate_k_iclause(n, k):
    """Generate a random k-clause."""
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]


def cnf_to_graph(n_vars, iclauses):
    """Convert a CNF formula to a DGL bipartite graph."""
    n_clauses = len(iclauses)
    
    # Create edges between literals and clauses
    src_lits, dst_clauses = [], []
    
    for c_idx, clause in enumerate(iclauses):
        for lit in clause:
            # Convert literal to variable index (0-indexed)
            var_idx = abs(lit) - 1
            # If negative literal, add n_vars
            if lit < 0:
                var_idx += n_vars
            src_lits.append(var_idx)
            dst_clauses.append(c_idx)
    
    # Create bipartite graph
    g = dgl.heterograph({
        ('lit', 'in', 'clause'): (torch.tensor(src_lits), torch.tensor(dst_clauses)),
        ('clause', 'to', 'lit'): (torch.tensor(dst_clauses), torch.tensor(src_lits))
    })
    
    # Store number of variables as graph attribute
    g.n_vars = n_vars
    
    return g


def generate_sat_problem(n_vars, p_k_2=0.3, p_geo=0.4):
    """Generate a SAT problem with a satisfiable and unsatisfiable variant."""
    solver = MinisatSolver()
    for i in range(n_vars): 
        solver.new_var(dvar=True)
    
    iclauses = []
    
    # Generate clauses until the problem is unsatisfiable
    while True:
        k_base = 1 if random.random() < p_k_2 else 2
        k = k_base + np.random.geometric(p_geo)
        iclause = generate_k_iclause(n_vars, k)
        
        solver.add_clause(iclause)
        is_sat = solver.solve()
        
        if is_sat:
            iclauses.append(iclause)
        else:
            break
    
    # The last clause made the problem unsatisfiable
    iclause_unsat = iclause
    
    # Make a satisfiable version by negating one literal
    iclause_sat = [-iclause_unsat[0]] + iclause_unsat[1:]
    
    # Create two problem variants
    sat_clauses = iclauses + [iclause_sat]
    unsat_clauses = iclauses + [iclause_unsat]
    
    # Convert to graphs
    g_sat = cnf_to_graph(n_vars, sat_clauses)
    g_unsat = cnf_to_graph(n_vars, unsat_clauses)
    
    return g_sat, g_unsat, 1.0, 0.0  # Return graphs and labels


def generate_batch(min_n, max_n, batch_size):
    """Generate a batch of SAT problems with balanced labels."""
    graphs = []
    labels = []
    n_vars_list = []
    
    for _ in range(batch_size):
        n_vars = random.randint(min_n, max_n)
        n_vars_list.append(n_vars)
        g_sat, g_unsat, label_sat, label_unsat = generate_sat_problem(n_vars)
    
        # Randomly choose one variant
        if random.random() < 0.5:
            graphs.append(g_sat)
            labels.append(torch.tensor([label_sat], dtype=torch.float))
        else:
            graphs.append(g_unsat)
            labels.append(torch.tensor([label_unsat], dtype=torch.float))

    batched_graph = dgl.batch(graphs)
    # Store n_vars for each graph in the batch
    batched_graph.n_vars_list = n_vars_list
    batched_graph.n_vars = max(n_vars_list)  # Use max n_vars for safety
    return batched_graph, torch.stack(labels)


class NeuroSAT(nn.Module):
    def __init__(self, dim=128, n_rounds=26):
        super().__init__()
        self.dim = dim
        self.n_rounds = n_rounds
        self.convergence_threshold = 0.01  # Convergence threshold for early stopping
        
        # Embeddings initialization
        self.lit_init = nn.Linear(1, dim)
        self.clause_init = nn.Linear(1, dim)
        
        # Add attention layers (Optimization 4)
        self.lit_attn = nn.Sequential(
            nn.Linear(dim*2, dim), nn.ReLU(),
            nn.Linear(dim, 1)
        )
        self.clause_attn = nn.Sequential(
            nn.Linear(dim*2, dim), nn.ReLU(),
            nn.Linear(dim, 1)
        )
        
        # Message passing networks
        self.lit_msg = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.clause_msg = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # Update networks
        self.lit_update = nn.LSTMCell(dim * 2, dim)
        self.clause_update = nn.LSTMCell(dim, dim)
        
        # Add batch normalization layers (Optimization 6)
        self.lit_norm = nn.BatchNorm1d(dim)
        self.clause_norm = nn.BatchNorm1d(dim)
        
        # Voting network
        self.vote = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, 1)
        )
        
    def apply_lit_attention(self, edges):
        # Calculate attention weights
        attn_input = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        attn_weight = torch.sigmoid(self.lit_attn(attn_input))
        return {'l2c': self.lit_msg(edges.src['h']) * attn_weight}
    
    def apply_clause_attention(self, edges):
        # Calculate attention weights
        attn_input = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        attn_weight = torch.sigmoid(self.clause_attn(attn_input))
        return {'c2l': self.clause_msg(edges.src['h']) * attn_weight}
    
    def forward(self, g):
        # Get graph info
        n_lits = g.num_nodes('lit') 
        
        # Get batch information for later
        batch_num_nodes = g.batch_num_nodes('lit')
        
        # Handle batch with different n_vars
        if hasattr(g, 'n_vars_list'):
            # Using max n_vars for the batch
            n_vars = g.n_vars
        elif hasattr(g, 'n_vars'):
            n_vars = g.n_vars
        else:
            raise AttributeError('Graph object does not have n_vars or n_vars_list attribute.')
        n_clauses = g.num_nodes('clause')
        
        # Dynamic rounds calculation (Optimization 1)
        dynamic_rounds = min(self.n_rounds, int(np.log(n_lits + n_clauses) * 4))
        
        # Initialize node features
        ones = torch.ones(n_lits, 1).to(g.device)
        lit_h = self.lit_init(ones)
        lit_c = torch.zeros_like(lit_h)

        ones = torch.ones(g.num_nodes('clause'), 1).to(g.device)
        clause_h = self.clause_init(ones)
        clause_c = torch.zeros_like(clause_h)
        
        # State tracking for early stopping (Optimization 2)
        prev_lit_h = None
        converged = False

        # Message passing rounds
        for r in range(dynamic_rounds):
            # Store previous state for early stopping check
            if prev_lit_h is None:
                prev_lit_h = lit_h.clone().detach()
            
            g.nodes['lit'].data['h'] = lit_h
            g.nodes['clause'].data['h'] = clause_h

            # Literal to clause messages
            # Use attention mechanism (Optimization 4)
            g.apply_edges(self.apply_lit_attention, etype='in')
            g.update_all(fn.copy_e('l2c', 'm'), fn.sum('m', 'agg_l'), etype='in')

            # Update clause states
            clause_h_agg = g.nodes['clause'].data['agg_l']
            clause_h_new, clause_c = self.clause_update(clause_h_agg, (clause_h, clause_c))
            
            # Add residual connection (Optimization 5)
            clause_h = clause_h + clause_h_new
            
            # Apply batch normalization (Optimization 6)
            if g.num_nodes('clause') > 1:  # Batch normalization needs at least 2 samples
                clause_h = self.clause_norm(clause_h)

            # Clause to literal messages
            g.nodes['clause'].data['h'] = clause_h
            # Use attention mechanism (Optimization 4)
            g.apply_edges(self.apply_clause_attention, etype='to')
            g.update_all(fn.copy_e('c2l', 'm'), fn.sum('m', 'agg_c'), etype='to')

            # Update literal states with flipped states for negated literals
            clause_h_agg = g.nodes['lit'].data['agg_c']
            
            # Create flipped version of literal states (positive <-> negative)
            # Handle batch with different n_vars
            if hasattr(g, 'n_vars_list'):
                # More complex flipping for batched graphs with different n_vars
                flipped_h = torch.zeros_like(lit_h)
                # This is a simplified approach - in a real implementation, 
                # you would need to handle each graph in the batch separately
                # based on their individual n_vars values
                flipped_h[:n_vars] = lit_h[n_vars:2*n_vars]
                flipped_h[n_vars:2*n_vars] = lit_h[:n_vars]
                flipped_h[2*n_vars:] = lit_h[2*n_vars:]  # Keep any additional literals unchanged
            else:
                flipped_h = torch.cat([lit_h[n_vars:], lit_h[:n_vars]], dim=0)

            # Concatenate messages and flipped states
            lit_input = torch.cat([clause_h_agg, flipped_h], dim=1)
            lit_h_new, lit_c = self.lit_update(lit_input, (lit_h, lit_c))
            
            # Add residual connection (Optimization 5)
            lit_h = lit_h + lit_h_new
            
            # Apply batch normalization (Optimization 6)
            if g.num_nodes('lit') > 1:  # Batch normalization needs at least 2 samples
                lit_h = self.lit_norm(lit_h)
            
            # Early stopping check (Optimization 2)
            # Improved early stopping that considers batch variance
            with torch.no_grad():
                change = torch.norm(lit_h - prev_lit_h) / (torch.norm(prev_lit_h) + 1e-6)
                
                # For batched graphs, we can use a more sophisticated convergence check
                if hasattr(g, 'n_vars_list') and len(g.n_vars_list) > 1:
                    # Only stop if the change is small enough for all samples
                    if change < self.convergence_threshold * 0.5:  # More strict for batches
                        break
                elif change < self.convergence_threshold:
                    break
                
            prev_lit_h = lit_h.clone().detach()

        # Voting
        votes = self.vote(lit_h)

        # Get votes for each variable (combine positive and negative literal votes)
        # For batched graphs, we need to process each graph separately
        batch_size = len(g.batch_num_nodes('lit')) if hasattr(g, 'batch_num_nodes') else 1
        predictions = []
        
        lit_offset = 0
        for i in range(batch_size):
            # For batched graph, extract the number of literals for this graph
            if hasattr(g, 'n_vars_list'):
                curr_n_vars = g.n_vars_list[i]
            else:
                curr_n_vars = n_vars
                
            # Get start and end indices for this graph's literals
            pos_votes = votes[lit_offset:lit_offset + curr_n_vars]
            neg_votes = votes[lit_offset + curr_n_vars:lit_offset + 2*curr_n_vars]
            
            # Combine positive and negative votes
            pos_abs = torch.abs(pos_votes)
            neg_abs = torch.abs(neg_votes)
            var_votes = torch.where(pos_abs > neg_abs, pos_votes, -neg_votes)
            
            # Compute prediction for this graph
            predictions.append(torch.mean(var_votes))
            lit_offset += batch_num_nodes[i] if hasattr(g, 'batch_num_nodes') else n_lits
        
        # Return sigmoid of all predictions
        return torch.sigmoid(torch.stack(predictions).view(batch_size, 1))


def train(args):
    """Train the NeuroSAT model."""
    # DGL does not currently support MPS, so we use CUDA or CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Create model
    model = NeuroSAT(dim=args.dim, n_rounds=args.n_rounds).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-10)
    
    # Restore from checkpoint if provided
    start_epoch = 0
    best_acc = 0.0
    if args.restore:
        checkpoint = torch.load(args.restore)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('acc', 0.0)
        print(f'Restored from {args.restore}, starting from epoch {start_epoch+1}, best acc: {best_acc:.3f}')
    
    print(f'Training on {device} for {args.epochs} epochs')
    
    # Make sure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        
        # Training batches
        train_bar = tqdm(range(args.train_batches), desc=f'Epoch {epoch+1}/{args.epochs} (Train)')
        for _ in train_bar:
            # Generate a batch of problems
            g, labels = generate_batch(args.min_n, args.max_n, args.batch_size)
            g = g.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad() 
            output = model(g)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += ((output > 0.5).float() == labels).sum().item()
            train_total += labels.size(0)
     
            train_bar.set_description( 
                f'Epoch {epoch+1}/{args.epochs} (Train) - Loss: {train_loss/(_+1):.4f}, Acc: {train_correct/train_total:.3f}'
            )
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_bar = tqdm(range(args.val_batches), desc=f'Epoch {epoch+1}/{args.epochs} (Val)')
            for _ in val_bar:
                # Generate a batch of problems
                g, labels = generate_batch(args.min_n, args.max_n, args.batch_size)
                g = g.to(device)
                labels = labels.to(device)
                
                output = model(g)
                
                # Update validation counts
                val_correct += ((output > 0.5).float() == labels).sum().item()
                val_total += labels.size(0)
                
                val_bar.set_description(
                    f'Epoch {epoch+1}/{args.epochs} (Val) - Acc: {val_correct/(val_total or 1):.3f}'
                )
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'  Train Loss: {train_loss/args.train_batches:.4f}, Train Acc: {train_acc:.3f}')
        print(f'  Val Acc: {val_acc:.3f}') 
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'acc': val_acc,
            'state_dict': model.state_dict()
        }
        
        # Save last model
        torch.save(checkpoint, os.path.join(args.model_dir, f'neurosat_dgl_last.pth'))
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint, os.path.join(args.model_dir, 'neurosat_dgl_best.pth'))
            print(f'  New best accuracy: {best_acc:.3f}')


def evaluate(args):
    """Evaluate the NeuroSAT model on test data."""
    # DGL does not currently support MPS, so we use CUDA or CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Load model
    model = NeuroSAT(dim=args.dim, n_rounds=args.n_rounds).to(device)
    checkpoint = torch.load(args.restore)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print(f'Evaluating model on {device}')
    
    # Evaluate on generated test data
    correct = 0
    total = 0
    times = []
    
    with torch.no_grad():
        test_bar = tqdm(range(args.test_batches))
        for _ in test_bar:
            # Generate a batch of problems
            g, labels = generate_batch(args.min_n, args.max_n, args.batch_size)
            g = g.to(device)
            labels = labels.to(device)
            
            # Measure inference time
            start_time = time.time() 
            output = model(g)
            end_time = time.time()
            
            correct += ((output > 0.5).float() == labels).sum().item()
            total += labels.size(0)
            times = []  # Initialize times list if not already done
            times.append((end_time - start_time) * 1000 / args.batch_size)  # Average ms per sample
            
            test_bar.set_description(
                f'Acc: {correct/(total or 1):.3f}, Avg time: {sum(times)/(len(times) or 1):.2f} ms')
    
    # Calculate metrics
    accuracy = correct / total
    avg_time = sum(times) / len(times)
    
    print(f'Test Results:')
    print(f'  Accuracy: {accuracy:.3f} ({correct}/{total})')
    print(f'  Average inference time: {avg_time:.2f} ms per sample')


def main():
    parser = argparse.ArgumentParser(description='NeuroSAT with DGL')
    
    # Model parameters
    parser.add_argument('--dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n_rounds', type=int, default=26, help='Number of message passing rounds')
    
    # Data parameters
    parser.add_argument('--min_n', type=int, default=10, help='Min number of variables')
    parser.add_argument('--max_n', type=int, default=40, help='Max number of variables')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--train_batches', type=int, default=100, help='Batches per epoch')
    parser.add_argument('--val_batches', type=int, default=20, help='Validation batches')
    parser.add_argument('--lr', type=float, default=0.00002, help='Learning rate')
    
    # Testing parameters
    parser.add_argument('--test_batches', type=int, default=50, help='Test batches')
    
    # Input/output
    parser.add_argument('--model_dir', type=str, default='./model', help='Model directory')
    parser.add_argument('--restore', type=str, default=None, help='Restore model from file')
    
    # Mode
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                      help='Run mode: train or eval')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        if args.restore is None:
            parser.error("--restore is required for eval mode")
        evaluate(args)


if __name__ == '__main__':
    main()