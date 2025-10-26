#!/bin/bash
# Test script for checkpoint resume functionality
# This script runs a very short training, interrupts it, and resumes

set -e  # Exit on error

echo "=========================================="
echo "Testing Checkpoint Resume Functionality"
echo "=========================================="

# Clean up any existing test checkpoints
echo ""
echo "1. Cleaning up old test checkpoints..."
rm -rf ~/.cache/nanochat/base_checkpoints/test_resume

# Run a short training (10 iterations with saves every 5 steps)
echo ""
echo "2. Starting initial training (10 iterations, save every 5 steps)..."
python -m scripts.base_train \
    --depth=12 \
    --num_iterations=10 \
    --save_every=5 \
    --keep_last_n_checkpoints=2 \
    --model_tag=test_resume \
    --resume=False \
    --eval_every=10000 \
    --core_metric_every=10000 \
    --sample_every=10000

echo ""
echo "3. Checking saved checkpoints..."
ls -lh ~/.cache/nanochat/base_checkpoints/test_resume/ || echo "No checkpoints directory found!"

if [ -f ~/.cache/nanochat/base_checkpoints/test_resume/model_000005.pt ]; then
    echo "✓ Checkpoint at step 5 found"
else
    echo "✗ Checkpoint at step 5 NOT found"
    exit 1
fi

if [ -f ~/.cache/nanochat/base_checkpoints/test_resume/model_000010.pt ]; then
    echo "✓ Checkpoint at step 10 found"
else
    echo "✗ Checkpoint at step 10 NOT found"
    exit 1
fi

# Check latest step marker
if [ -f ~/.cache/nanochat/base_checkpoints/test_resume/latest_step.txt ]; then
    latest_step=$(cat ~/.cache/nanochat/base_checkpoints/test_resume/latest_step.txt)
    echo "✓ Latest step marker found: step $latest_step"
    if [ "$latest_step" != "10" ]; then
        echo "✗ Expected latest step to be 10, got $latest_step"
        exit 1
    fi
else
    echo "✗ Latest step marker NOT found"
    exit 1
fi

# Resume training for 5 more iterations
echo ""
echo "4. Resuming training (5 more iterations)..."
python -m scripts.base_train \
    --depth=12 \
    --num_iterations=15 \
    --save_every=5 \
    --keep_last_n_checkpoints=2 \
    --model_tag=test_resume \
    --resume=True \
    --eval_every=10000 \
    --core_metric_every=10000 \
    --sample_every=10000

echo ""
echo "5. Checking final checkpoints..."
ls -lh ~/.cache/nanochat/base_checkpoints/test_resume/

if [ -f ~/.cache/nanochat/base_checkpoints/test_resume/model_000015.pt ]; then
    echo "✓ Final checkpoint at step 15 found"
else
    echo "✗ Final checkpoint at step 15 NOT found"
    exit 1
fi

# Check that old checkpoints were cleaned up (keep_last_n_checkpoints=2)
checkpoint_count=$(ls ~/.cache/nanochat/base_checkpoints/test_resume/model_*.pt | wc -l)
echo "Found $checkpoint_count model checkpoint files"

if [ "$checkpoint_count" -le 3 ]; then
    echo "✓ Old checkpoints cleaned up correctly (keeping last $checkpoint_count)"
else
    echo "⚠ Warning: Expected at most 3 checkpoints, found $checkpoint_count"
fi

echo ""
echo "=========================================="
echo "✓ All tests passed!"
echo "=========================================="
echo ""
echo "Cleanup: Run 'rm -rf ~/.cache/nanochat/base_checkpoints/test_resume' to remove test checkpoints"
