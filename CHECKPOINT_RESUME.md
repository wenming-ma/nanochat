# Checkpoint Resume Feature

## Overview

Nanochat now supports automatic checkpoint saving and resume from interruptions during training.

## New Configuration Parameters

### In `scripts/base_train.py` (and will be added to other training scripts):

```python
save_every = 500              # Save checkpoint every N steps (0 = only save at end)
keep_last_n_checkpoints = 3   # Keep only last N checkpoints to save disk space (-1 = keep all)
resume = True                 # Automatically resume from latest checkpoint if available
```

## Features

### 1. Periodic Checkpoint Saving

- Checkpoints are now saved every `save_every` steps (default: 500)
- Each checkpoint includes:
  - Model state (weights)
  - Optimizer state (both AdamW and Muon)
  - Training metadata (step, validation loss, config)

### 2. Automatic Resume

- On training start, the script checks for existing checkpoints in the checkpoint directory
- If found and `resume=True`, it will:
  - Load the latest checkpoint
  - Restore model weights
  - Restore optimizer states
  - Continue training from the next step

### 3. Disk Space Management

- Automatically removes old checkpoints, keeping only the last N (default: 3)
- This prevents disk space issues during long training runs
- Set `keep_last_n_checkpoints=-1` to keep all checkpoints

### 4. Latest Step Marker

- Creates a `latest_step.txt` file in the checkpoint directory
- Makes it easy to find the most recent checkpoint

## Usage Examples

### Basic Usage (with defaults)

```bash
# Start training (or resume if checkpoint exists)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
```

### Disable Auto-Resume

```bash
# Start fresh training even if checkpoints exist
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --resume=False
```

### Custom Save Frequency

```bash
# Save every 1000 steps instead of 500
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --save_every=1000
```

### Keep All Checkpoints

```bash
# Don't delete old checkpoints
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --keep_last_n_checkpoints=-1
```

### Save Only at End (Original Behavior)

```bash
# Disable periodic saves
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --save_every=0
```

## How It Works

### Training Flow

1. **Training Start**:
   - Script checks for existing checkpoint directory
   - If `resume=True` and checkpoints exist, loads the latest one
   - Sets `start_step` to resume from the correct iteration

2. **During Training**:
   - Every `save_every` steps, saves a checkpoint
   - After saving, removes old checkpoints if `keep_last_n_checkpoints > 0`
   - Updates `latest_step.txt` file

3. **Training End**:
   - Saves final checkpoint at the last step
   - Final checkpoint is always kept (not removed by cleanup)

### Checkpoint Files

Each checkpoint consists of 3 files:
- `model_XXXXXX.pt` - Model weights
- `optim_XXXXXX.pt` - Optimizer state
- `meta_XXXXXX.json` - Metadata (step, config, metrics)

Where `XXXXXX` is the zero-padded step number (e.g., `000500`, `001000`)

### Recovery Scenarios

#### Power Failure / OOM / Crash

If training is interrupted at step 2750 (and `save_every=500`):
- Latest checkpoint will be at step 2500
- Resume will start from step 2501
- You lose only 250 steps of progress (not all 2750)

#### Manual Interruption (Ctrl+C)

Same as above - resume from the last saved checkpoint.

#### Out of Disk Space

- The cleanup mechanism prevents this by removing old checkpoints
- If you need more checkpoints, increase `keep_last_n_checkpoints`

## Benefits

✅ **Fault Tolerance**: Training can recover from interruptions
✅ **Time Savings**: Don't lose hours of training progress
✅ **Disk Management**: Automatic cleanup of old checkpoints
✅ **Flexible**: Configure save frequency based on your needs
✅ **Backward Compatible**: Can disable to get original behavior

## Limitations

- Resume continues from the last checkpoint step, not the exact interrupted step
- If `save_every=500`, you may lose up to 499 steps of progress
- Trade-off: More frequent saves = less progress loss but more I/O overhead

## Recommended Settings

### For Speedrun (d20, ~4 hours)
```bash
--save_every=500 --keep_last_n_checkpoints=3
```
Saves every ~10-15 minutes, keeps last 3 checkpoints (~1.5-2GB total)

### For Long Runs (d32, ~41 hours)
```bash
--save_every=1000 --keep_last_n_checkpoints=5
```
Saves every ~30-45 minutes, keeps last 5 checkpoints (~5-7GB total)

### For Debugging/Testing
```bash
--save_every=50 --keep_last_n_checkpoints=2
```
Saves frequently for testing, keeps only 2 recent checkpoints

## Future Improvements

Potential enhancements for future versions:
- [ ] Resume support for mid_train.py, chat_sft.py, chat_rl.py
- [ ] Checkpoint compression to reduce disk usage
- [ ] Cloud backup integration (S3, GCS, etc.)
- [ ] Resume with different hyperparameters (e.g., learning rate)
- [ ] Multi-checkpoint ensemble evaluation
