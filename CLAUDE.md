# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nanochat is a full-stack LLM implementation designed to train ChatGPT-like models from scratch on a single 8xH100 GPU node in 4-41 hours for $100-$1000. It includes the complete pipeline: tokenization, pretraining, midtraining, supervised finetuning, reinforcement learning, evaluation, and web serving.

Key characteristics:
- Minimal, hackable codebase (~8,000 lines across 45 files)
- Dependency-light with custom Rust tokenizer
- Designed to fit entirely in an LLM context window
- Single cohesive codebase (not a framework)

## Development Environment Setup

### Python Environment
```bash
# Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

### Rust Tokenizer Build
```bash
# Install Rust/Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build rustbpe tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## Common Development Commands

### Training Pipeline
```bash
# Full end-to-end training (d20, ~4 hours, $100)
bash speedrun.sh

# Larger model (d32, ~41 hours, $1000)
bash run1000.sh

# Run in screen session with logging
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Individual training stages (multi-GPU)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl  # Optional

# Single GPU training (8x slower, automatic gradient accumulation)
python -m scripts.base_train --depth=20
python -m scripts.mid_train
python -m scripts.chat_sft
python -m scripts.chat_rl  # Optional
```

### Tokenizer
```bash
# Train tokenizer
python -m scripts.tok_train --max_chars=2000000000

# Evaluate tokenizer
python -m scripts.tok_eval
```

### Evaluation
```bash
# Base model evaluation
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# Chat model evaluation (specify checkpoint: mid/sft/rl)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# Evaluate specific task only
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K
```

### Inference
```bash
# CLI chat (interactive)
python -m scripts.chat_cli

# CLI chat (single prompt)
python -m scripts.chat_cli -p "Why is the sky blue?"

# Web UI (ChatGPT-like interface)
python -m scripts.chat_web
# Then visit http://localhost:8000 or http://<public-ip>:8000
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_rustbpe.py -v -s

# Skip slow tests
python -m pytest -m "not slow"
```

### Report Generation
```bash
# Reset report (clears previous run data)
python -m nanochat.report reset

# Generate final report (aggregates all metrics)
python -m nanochat.report generate
# Output: report.md in current directory
```

### Data Management
```bash
# Download data shards (n = number of shards)
python -m nanochat.dataset -n 240

# Data is cached to ~/.cache/nanochat/ by default
# Override with: export NANOCHAT_BASE_DIR=/custom/path
```

## Architecture Overview

### Training Pipeline Flow

The complete pipeline follows this sequence:

1. **Tokenizer Training**: Build custom BPE tokenizer with vocab size 65,536 on ~2B characters
2. **Pretraining (BASE)**: Train base model on 2-38B tokens from HuggingFace fineweb-edu-100b
3. **Midtraining (MID)**: Teach special tokens, conversation structure, and tool use
4. **Supervised Finetuning (SFT)**: Domain adaptation to chat format with lower learning rates
5. **Reinforcement Learning (RL)**: Optional reward-based training on specific tasks (GSM8K)
6. **Evaluation**: Multi-task benchmarking (CORE, ARC, MMLU, HumanEval, GSM8K, ChatCORE)
7. **Serving**: Web UI or CLI inference with KV-cache optimization

### Core Components

**nanochat/ package** (15 modules):
- `gpt.py`: GPT transformer with modern architecture (rotary embeddings, QK norm, MQA, ReLU², RMSNorm)
- `engine.py`: Inference engine with KV-cache, token streaming, calculator tools
- `tokenizer.py`: BPE tokenizer wrapper (dual mode: HuggingFace for training, tiktoken/rustbpe for inference)
- `dataset.py`: On-demand parquet shard downloading from HuggingFace
- `dataloader.py`: Distributed streaming data loader with tokenization
- `checkpoint_manager.py`: Model/optimizer checkpointing and loading
- `loss_eval.py`: Bits-per-byte (BpB) metric evaluation
- `core_eval.py`: CORE metric evaluation (from DCLM paper)
- `configurator.py`: CLI argument parser for hyperparameter overrides
- `common.py`: Distributed setup, logging, base directory management
- `muon.py`: Muon optimizer (for matrix parameters)
- `adamw.py`: AdamW optimizer (for embedding/unembedding parameters)
- `execution.py`: Gradient accumulation and training loop
- `report.py`: Report generation with metrics and system info

**scripts/** (11 entry points):
- Training: `tok_train.py`, `base_train.py`, `mid_train.py`, `chat_sft.py`, `chat_rl.py`
- Evaluation: `tok_eval.py`, `base_loss.py`, `base_eval.py`, `chat_eval.py`
- Inference: `chat_cli.py`, `chat_web.py`

**tasks/** (7 evaluation tasks):
- `smoltalk.py`: Conversation dataset
- `gsm8k.py`: Math reasoning (generative)
- `mmlu.py`: Multiple choice knowledge
- `arc.py`: Science reasoning (ARC-Easy, ARC-Challenge)
- `humaneval.py`: Code generation
- `customjson.py`: Custom JSONL conversations
- `common.py`: Base classes (`Task`, `TaskMixture`)

**rustbpe/**: Rust-based BPE tokenizer implementation (compiled to Python extension via maturin)

### Model Architecture

**Modern GPT design features**:
- Rotary positional embeddings (no learnable pos embeddings)
- QK normalization in attention
- Multi-Query Attention (MQA) for efficient inference
- ReLU² activation in MLPs
- RMSNorm with no learnable parameters
- No bias in linear layers
- Untied token/unembedding weights

**Model scaling** (controlled by `--depth` parameter):
- `depth=20` (d20): 561M params, ~4 hours training (speedrun default)
- `depth=26` (d26): ~1B params, ~12 hours training (GPT-2 grade)
- `depth=32` (d32): 1.9B params, ~41 hours training (run1000 default)
- Model dim = `depth * 64`
- Num heads derived to target head_dim=128

### Optimizer Configuration

**Layered learning rates** (4 separate parameter groups):
- Embedding parameters: lr=0.2 (AdamW)
- Unembedding parameters: lr=0.004 (AdamW)
- Matrix parameters: lr=0.02 (Muon specialized 2nd-order optimizer)
- Weight decay: 0.0, Gradient clipping: 1.0

**Batch sizing**:
- Global batch size: 524,288 tokens (fixed)
- Device batch size: ~32 for H100 80GB (tune with `--device_batch_size` to avoid OOM)
- Automatic gradient accumulation compensation when device batch size reduced

### Data Pipeline

**Source**: HuggingFace fineweb-edu-100b
- 1,822 parquet shards, ~250M characters each
- On-demand download to `~/.cache/nanochat/`
- Distributed streaming with proper rank sharding

**Chinchilla scaling** (20 tokens per parameter):
- d20 (561M params): 11.2B tokens → 240 shards needed
- d32 (1.9B params): 37.6B tokens → 800 shards needed

### Evaluation Metrics

**Pretraining**:
- Bits-per-byte (BpB): Vocab-size independent loss metric
- CORE: Multi-task evaluation from DCLM paper

**Finetuning**:
- Generative tasks: Greedy decoding with human-readable comparisons
- Multiple choice: Accuracy on MMLU, ARC-Easy, ARC-Challenge
- Code: HumanEval pass rate
- Math: GSM8K accuracy with chain-of-thought

Output: `report.md` with summary table across all training phases (BASE/MID/SFT/RL)

## Important Development Notes

### Hyperparameter Tuning

All scripts use `configurator.py` for CLI argument parsing. Override defaults without code changes:

```bash
# Increase model size (requires more shards and lower batch size to fit in memory)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16

# Adjust learning rates
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --lr_emb=0.3 --lr_unemb=0.005

# Change max sequence length
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --max_seqlen=2048
```

### Memory Management

If you encounter OOM errors:
1. Reduce `--device_batch_size` (32 → 16 → 8 → 4 → 2 → 1)
2. Reduce model size with `--depth` parameter
3. Code automatically compensates with gradient accumulation

### Distributed Training

- Default: Uses `torchrun` with 8 GPUs (DDP)
- Single GPU: Omit `torchrun` (automatic gradient accumulation, 8x slower)
- The code uses gradient accumulation to maintain effective batch size regardless of device count

### Personality Customization

To customize your nanochat's personality:
1. Modify synthetic identity conversations: see `dev/gen_sft_data.py`
2. Replace `identity_conversations.jsonl` downloaded in speedrun.sh
3. The identity is injected during midtraining via `tasks/customjson.py`

See: [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139)

### WandB Integration

Optional but recommended for metrics tracking:

```bash
# Login to wandb first
wandb login

# Set WANDB_RUN environment variable
WANDB_RUN=my_experiment bash speedrun.sh
```

Without `WANDB_RUN`, defaults to "dummy" (no wandb logging)

### Data Requirements

Training data automatically downloaded from HuggingFace during training. To pre-download:

```bash
# For d20 speedrun (240 shards, ~24GB)
python -m nanochat.dataset -n 240

# For d32 run1000 (800 shards, ~80GB)
python -m nanochat.dataset -n 800
```

Shards are cached to `$NANOCHAT_BASE_DIR` (default: `~/.cache/nanochat/`)

### File Locations

- Checkpoints: `$NANOCHAT_BASE_DIR/{base,mid,sft,rl}/` directories
- Data shards: `$NANOCHAT_BASE_DIR/data/`
- Tokenizer: `$NANOCHAT_BASE_DIR/tok_65536.model`
- Eval bundle: `$NANOCHAT_BASE_DIR/eval_bundle/`
- Identity data: `$NANOCHAT_BASE_DIR/identity_conversations.jsonl`
- Report: `$NANOCHAT_BASE_DIR/report/` (sections assembled into `report.md`)

### Running on Different Hardware

**8xA100 (vs 8xH100)**:
- Works out of the box, slightly slower training
- No hyperparameter changes needed

**Single GPU**:
- Omit `torchrun` from commands
- Training will be 8x slower (sequential gradient accumulation)
- Results will be nearly identical

**Less than 80GB VRAM**:
- Reduce `--device_batch_size` until it fits
- Start with 16, then 8, 4, 2, 1 if needed

**CPU / MPS (Mac)**:
- See [CPU|MPS PR #88](https://github.com/karpathy/nanochat/pull/88)
- Use `--device_type=mps` for Mac GPUs
- Not practical for large-scale training, useful for code exploration

## Code Philosophy

Nanochat is intentionally:
- **Minimal**: No giant config objects or model factories
- **Readable**: Entire codebase fits in LLM context (~330KB, ~100K tokens)
- **Hackable**: Direct code over abstractions; easy to fork and modify
- **Single-purpose**: Not a framework; a concrete ChatGPT implementation

**NOT exhaustively configurable**: The codebase prioritizes clarity and simplicity over flexibility. It's a "strong baseline" designed to be understood and modified, not a production framework.

## Additional Resources

- Main discussion: ["Introducing nanochat: The best ChatGPT that $100 can buy"](https://github.com/karpathy/nanochat/discussions/1)
- Personality customization: [Guide in Discussions #139](https://github.com/karpathy/nanochat/discussions/139)
- Package repo with files-to-prompt: `files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt`
- Ask questions via [DeepWiki](https://deepwiki.com/): Replace `github.com` with `deepwiki.com` in repo URL
