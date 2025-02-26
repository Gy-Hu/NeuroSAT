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
            labels.append(label_sat)
        else:
            graphs.append(g_unsat)
            labels.append(label_unsat)

    batched_graph = dgl.batch(graphs)
    # Set n_vars attribute for the batched graph
    batched_graph.n_vars = graphs[0].n_vars
    return batched_graph, torch.stack(labels)


class NeuroSAT(nn.Module):
    def __init__(self, dim=128, n_rounds=26):
        super().__init__()
        self.dim = dim
        self.n_rounds = n_rounds
        self.convergence_threshold = 0.01  # 早停机制的收敛阈值
        
        # Embeddings initialization
        self.lit_init = nn.Linear(1, dim)
        self.clause_init = nn.Linear(1, dim)
        
        # 添加注意力层 (优化4)
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
        
        # 添加批归一化层 (优化6)
        self.lit_norm = nn.BatchNorm1d(dim)
        self.clause_norm = nn.BatchNorm1d(dim)
        
        # Voting network
        self.vote = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, 1)
        )
        
    def apply_lit_attention(self, edges):
        # 计算注意力权重
        attn_input = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        attn_weight = torch.sigmoid(self.lit_attn(attn_input))
        return {'l2c': self.lit_msg(edges.src['h']) * attn_weight}
    
    def apply_clause_attention(self, edges):
        # 计算注意力权重
        attn_input = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        attn_weight = torch.sigmoid(self.clause_attn(attn_input))
        return {'c2l': self.clause_msg(edges.src['h']) * attn_weight}
    
    def forward(self, g):
        # Get graph info
        n_lits = g.num_nodes('lit')
        if not hasattr(g, 'n_vars'):
            raise AttributeError('Graph object does not have n_vars attribute.')
        n_vars = g.n_vars
        n_clauses = g.num_nodes('clause')
        
        # 动态轮数计算 (优化1)
        dynamic_rounds = min(self.n_rounds, int(np.log(n_lits + n_clauses) * 4))
        
        # Initialize node features
        ones = torch.ones(n_lits, 1).to(g.device)
        lit_h = self.lit_init(ones)
        lit_c = torch.zeros_like(lit_h)

        ones = torch.ones(g.num_nodes('clause'), 1).to(g.device)
        clause_h = self.clause_init(ones)
        clause_c = torch.zeros_like(clause_h)
        
        # 早停机制的状态跟踪 (优化2)
        prev_lit_h = None
        converged = False

        # Message passing rounds
        for r in range(dynamic_rounds):
            # 存储前一轮状态用于早停检查
            if prev_lit_h is None:
                prev_lit_h = lit_h.clone().detach()
            
            g.nodes['lit'].data['h'] = lit_h
            g.nodes['clause'].data['h'] = clause_h

            # Literal to clause messages
            # 使用注意力机制 (优化4)
            g.apply_edges(self.apply_lit_attention, etype='in')
            g.update_all(fn.copy_e('l2c', 'm'), fn.sum('m', 'agg_l'), etype='in')

            # Update clause states
            clause_h_agg = g.nodes['clause'].data['agg_l']
            clause_h_new, clause_c = self.clause_update(clause_h_agg, (clause_h, clause_c))
            
            # 添加残差连接 (优化5)
            clause_h = clause_h + clause_h_new
            
            # 应用批归一化 (优化6)
            if g.num_nodes('clause') > 1:  # 批归一化需要至少2个样本
                clause_h = self.clause_norm(clause_h)

            # Clause to literal messages
            g.nodes['clause'].data['h'] = clause_h
            # 使用注意力机制 (优化4)
            g.apply_edges(self.apply_clause_attention, etype='to')
            g.update_all(fn.copy_e('c2l', 'm'), fn.sum('m', 'agg_c'), etype='to')

            # Update literal states with flipped states for negated literals
            clause_h_agg = g.nodes['lit'].data['agg_c']
            
            # Create flipped version of literal states (positive <-> negative)
            flipped_h = torch.cat([lit_h[n_vars:], lit_h[:n_vars]], dim=0)

            # Concatenate messages and flipped states
            lit_input = torch.cat([clause_h_agg, flipped_h], dim=1)
            lit_h_new, lit_c = self.lit_update(lit_input, (lit_h, lit_c))
            
            # 添加残差连接 (优化5)
            lit_h = lit_h + lit_h_new
            
            # 应用批归一化 (优化6)
            if g.num_nodes('lit') > 1:  # 批归一化需要至少2个样本
                lit_h = self.lit_norm(lit_h)
            
            # 早停检查 (优化2)
            change = torch.norm(lit_h - prev_lit_h) / (torch.norm(prev_lit_h) + 1e-6)
            if change < self.convergence_threshold:
                break
                
            prev_lit_h = lit_h.clone().detach()

        # Voting
        votes = self.vote(lit_h)

        # Get votes for each variable (combine positive and negative literal votes)
        pos_votes = votes[:n_vars]
        neg_votes = votes[n_vars:2*n_vars]
        var_votes = (pos_votes + neg_votes) / 2

        # Output a scalar prediction by taking the mean of all variable votes
        prediction = torch.mean(var_votes)
        return torch.sigmoid(prediction.view(1, 1))


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
            # Generate individual problems and process one by one
            batch_loss = 0.0
            batch_correct = 0
            batch_total = 0
            
            for _ in range(args.batch_size):
                n_vars = random.randint(args.min_n, args.max_n)
                g_sat, g_unsat, label_sat, label_unsat = generate_sat_problem(n_vars)
                
                # Randomly choose one variant
                if random.random() < 0.5:
                    g = g_sat.to(device)
                    label = torch.tensor([[label_sat]], dtype=torch.float32).to(device)
                else:
                    g = g_unsat.to(device)
                    label = torch.tensor([[label_unsat]], dtype=torch.float32).to(device)
                
                optimizer.zero_grad() 
                output = model(g)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
                pred = (output > 0.5).float()
                batch_correct += (pred == label).sum().item()
                batch_total += 1
            
            train_loss += batch_loss / args.batch_size
            train_correct += batch_correct
            train_total += batch_total
     
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
                # Generate individual problems and process one by one
                batch_correct = 0
                batch_total = 0
                
                for _ in range(args.batch_size):
                    n_vars = random.randint(args.min_n, args.max_n)
                    g_sat, g_unsat, label_sat, label_unsat = generate_sat_problem(n_vars)
                    
                    # Randomly choose one variant
                    if random.random() < 0.5:
                        g = g_sat.to(device)
                        label = torch.tensor([[label_sat]], dtype=torch.float32).to(device)
                    else:
                        g = g_unsat.to(device)
                        label = torch.tensor([[label_unsat]], dtype=torch.float32).to(device)
                    
                    output = model(g)
                
                    # Calculate accuracy
                    pred = (output > 0.5).float()
                    batch_correct += (pred == label).sum().item()
                    batch_total += 1
                
                # Update validation counts
                val_correct += batch_correct
                val_total += batch_total
                
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
            batch_correct = 0
            batch_total = 0
            batch_times = []
            
            for _ in range(args.batch_size):
                n_vars = random.randint(args.min_n, args.max_n)
                g_sat, g_unsat, label_sat, label_unsat = generate_sat_problem(n_vars)
                
                # Randomly choose one variant
                if random.random() < 0.5:
                    g = g_sat.to(device)
                    label = torch.tensor([[label_sat]], dtype=torch.float32).to(device)
                else:
                    g = g_unsat.to(device)
                    label = torch.tensor([[label_unsat]], dtype=torch.float32).to(device)

                # Measure inference time
                start_time = time.time() 
                output = model(g)
                end_time = time.time()
                
                # Calculate accuracy
                pred = (output > 0.5).float() 
                batch_correct += (pred == label).sum().item()
                batch_total += 1
                
                # Record times
                batch_times.append((end_time - start_time) * 1000)  # Convert to ms
                
            correct += batch_correct
            total += batch_total
            times.extend(batch_times)
            
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