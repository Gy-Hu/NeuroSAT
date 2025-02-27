import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task-name', type=str, default='neurosat', help='task name')

parser.add_argument('--dim', type=int, default=128, help='Dimension of variable and clause embeddings')
parser.add_argument('--n_rounds', type=int, default=26, help='Number of rounds of message passing')
parser.add_argument('--epochs', type=int, default=10)

parser.add_argument('--n_pairs', action='store', type=int)
parser.add_argument('--max_nodes_per_batch', action='store', type=int, help='Maximum number of nodes per batch')
parser.add_argument('--auto_batch_size', action='store_true', help='Automatically adjust batch size based on problem size')
parser.add_argument('--small_batch_size', type=int, default=12000, help='Batch size for small problems (n <= 10)')
parser.add_argument('--medium_batch_size', type=int, default=6000, help='Batch size for medium problems (10 < n <= 40)')
parser.add_argument('--large_batch_size', type=int, default=3000, help='Batch size for large problems (n > 40)')
parser.add_argument('--disable_gc', action='store_true', help='Disable garbage collection for better performance when enough memory is available')
parser.add_argument('--gen_log', type=str, default='/Users/apple/coding_env/NeuroSAT/log/data_maker.log')
parser.add_argument('--min_n', type=int, default=10, help='min number of variables used for training')
parser.add_argument('--max_n', type=int, default=40, help='max number of variables used for training')
parser.add_argument('--p_k_2', action='store', dest='p_k_2', type=float, default=0.3)
parser.add_argument('--p_geo', action='store', dest='p_geo', type=float, default=0.4)
parser.add_argument('--py_seed', action='store', dest='py_seed', type=int, default=None)
parser.add_argument('--np_seed', action='store', dest='np_seed', type=int, default=None)
parser.add_argument('--one', action='store', dest='one', type=int, default=0)

parser.add_argument('--log-dir', type=str, default='/Users/apple/coding_env/NeuroSAT/log/', help='log folder dir')
parser.add_argument('--model-dir', type=str, default='/Users/apple/coding_env/NeuroSAT/model/', help='model folder dir')
parser.add_argument('--data-dir', type=str, default='/Users/apple/coding_env/NeuroSAT/data/', help='data folder dir')
parser.add_argument('--restore', type=str, default=None, help='continue train from model')

parser.add_argument('--train-file', type=str, default=None, help='train file dir')
parser.add_argument('--val-file', type=str, default=None, help='val file dir')
