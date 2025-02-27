- [x] Turn off the gabage collect such as `gc.xxx` and `torch.mps.empty_cache()` when has enough memory
- [x] Reduce the frequency of `to(self.device)`
- [x] Increasing the `max_nodes_per_batch` ---> fully utilize GPU

## Optimizations Implemented

### 1. Memory Management Optimizations
- Added `--disable_gc` flag to disable garbage collection when there's enough memory
- Reduced unnecessary calls to `gc.collect()` and `torch.mps.empty_cache()`
- Implemented conditional memory cleanup with `_maybe_collect_garbage()` method

### 2. Device Transfer Optimizations
- Reduced frequency of `.to(self.device)` calls by ensuring tensors are created on the correct device
- Eliminated redundant device transfers in the forward pass
- Improved the `flip` method to avoid unnecessary device transfers

### 3. Batch Size Optimizations
- Added `--auto_batch_size` flag to automatically adjust batch size based on problem size
- Added configurable batch sizes for different problem sizes:
  - `--small_batch_size`: For small problems (n ≤ 10)
  - `--medium_batch_size`: For medium problems (10 < n ≤ 40)
  - `--large_batch_size`: For large problems (n > 40)

## Usage Instructions

### For Small Problems (SR3-10)
The `run_sr3to10.sh` script is configured to use a larger batch size (12000) to fully utilize the GPU.

### For Medium to Large Problems
For medium to large problems, you can use the `--auto_batch_size` flag to automatically adjust the batch size based on the problem size, or manually set the `--max_nodes_per_batch` parameter.

### Example Command
```bash
python src/train.py \
  --auto_batch_size \
  --small_batch_size 12000 \
  --medium_batch_size 6000 \
  --large_batch_size 3000 \
  --disable_gc
```

### Memory Management
If you have enough GPU memory, use the `--disable_gc` flag to disable garbage collection for better performance.
