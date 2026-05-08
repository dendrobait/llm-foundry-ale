# Distributed Training

Production-ready distributed training scripts for large language models using PyTorch's DDP (Distributed Data Parallel) and FSDP (Fully Sharded Data Parallel) strategies.

## Overview

This folder contains distributed training scripts for large language models using PyTorch's DDP (Distributed Data Parallel) and FSDP (Fully Sharded Data Parallel) strategies. Both are optimized for multi-GPU, multi-node SLURM clusters and support standard AdamW or hybrid Muon + Adam optimizers. The scripts also support working with several types of different architectures (dense transformers, mixture of experts, hybrid models) and can be easily extended to new ones.

## Available Training Scripts

- **train_ddp.py** — Distributed Data Parallel (DDP) training script for transformer-based causal language models. Handles multi-GPU synchronization with gradient accumulation and checkpointing.
- **train_fsdp.py** — Fully Sharded Data Parallel (FSDP) training script for larger models requiring parameter and optimizer state sharding across nodes.

## Core Modules

- **trainer.py** — Contains DDPTrainer and FSDPTrainer classes that encapsulate the training and validation loops, checkpointing, and per-step logging.
- **model_setup.py** — Pre-DDP/FSDP model and tokenizer initialization, including architecture setup and optional context extension for continual pretraining.
- **data_loading.py** — Dataset loading and DataLoader creation with support for multiple data formats (JSONL, Parquet).
- **optimizers.py** — Optimizer and learning rate scheduler creation for both AdamW and Muon + Adam configurations.
- **mfu.py** — Model FLOPs Utilization (MFU) calculation utilities for performance monitoring and benchmarking.
- **specifications.py** — Dataclass definitions and type hints for all training arguments.
- **utils.py** — Logging, checkpointing, distributed environment setup, and miscellaneous utilities.

## Running Training

1. Configure `specifications.yaml` with your training settings:
   - Dataset paths (`train_dataset_dir`, `val_dataset_dir`, `checkpoint_dir`)
   - Model configuration (`path_to_model_config`, `base_model`, `tokenizer_name_or_path`)
   - Training parameters (`learning_rate`, `batch_size`, `num_train_steps`, etc.)
   - Optimization settings (optimizer choice, warmup, scheduler type)

2. Launch training with SLURM using the provided shell scripts:
   - DDP training: `python distributed/train_ddp.py --specs path/to/specifications.yaml --slurm-job-id $SLURM_JOB_ID --hardware "a100"`
   - FSDP training: `python distributed/train_fsdp.py --specs path/to/specifications.yaml --slurm-job-id $SLURM_JOB_ID --hardware "a100"`

   Or use the shell scripts for SLURM job submission:
   - See `train_ddp.sh` for DDP SLURM configuration
   - See `train_fsdp.sh` for FSDP SLURM configuration

## Example Architecture Configs

Here we have toy examples of model config files covering a range of architectures (and HF definitions) the codebase supports. Each config is a `transformers`-compatible JSON that can be passed directly to `path_to_model_config` in `specifications.yaml`.

- **NOTE**: For hybrid-mamba models (e.g., GraniteMoeHybridForCausalLM), if you want maximum performance, you will need to install the `mamba-ssm` package with the `causal-conv1d` extra to get the optimized CUDA kernels for the Mamba layers (i.e., `pip install mamba-ssm[causal-conv1d] --no-build-isolation`). For models that use linear forms of attention like delta rule or gated delta rule (e.g., OlmoHybridForCausalLM, Qwen3NextForCausalLM), installing the `flash-linear-attention` package will provide optimized kernels for the linear attention parts (i.e., `pip install flash-linear-attention`). If these packages are not installed, the training will still run, but will be significantly slower.

<details>
<summary><strong>Dense Transformer</strong> — <code>LlamaForCausalLM</code> · Dense transformer · <a href="https://huggingface.co/docs/transformers/model_doc/llama">HF Docs</a></summary>

```json
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "hidden_act": "silu",
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 1536,
  "is_llama_config": true,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 8,
  "num_hidden_layers": 8,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_interleaved": false,
  "rope_scaling": null,
  "rope_theta": 100000,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "vocab_size": 49152
}
```

</details>

<details>
<summary><strong>Mixture of Experts</strong> — <code>Qwen3MoeForCausalLM</code> · Mixture of Experts · <a href="https://huggingface.co/docs/transformers/model_doc/qwen3_moe">HF Docs</a></summary>

```json
{
  "architectures": [
    "Qwen3MoeForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "decoder_sparse_step": 1,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 1536,
  "max_position_embeddings": 4096,
  "max_window_layers": 8,
  "mlp_only_layers": [],
  "model_type": "qwen3_moe",
  "moe_intermediate_size": 384,
  "norm_topk_prob": true,
  "num_attention_heads": 8,
  "num_experts": 8,
  "num_experts_per_tok": 2,
  "num_hidden_layers": 8,
  "num_key_value_heads": 8,
  "output_router_logits": false,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 100000,
  "router_aux_loss_coef": 0.001,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 49152
}
```

</details>

<details>
<summary><strong>Linear-Attention Hybrid</strong> — <code>OlmoHybridForCausalLM</code> · Linear-attention hybrid · <a href="https://huggingface.co/docs/transformers/v5.8.0/en/model_doc/olmo_hybrid">HF Docs</a></summary>

```json
{
  "model_type": "olmo_hybrid",
  "architectures": [
    "OlmoHybridForCausalLM"
  ],
  "vocab_size": 49152,
  "hidden_size": 512,
  "intermediate_size": 1536,
  "num_hidden_layers": 8,
  "num_attention_heads": 8,
  "num_key_value_heads": 8,
  "hidden_act": "silu",
  "max_position_embeddings": 4096,
  "initializer_range": 0.02,
  "use_cache": true,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "rms_norm_eps": 1e-05,
  "tie_word_embeddings": true,
  "layer_types": [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention"
  ],
  "linear_num_key_heads": 8,
  "linear_num_value_heads": 8,
  "linear_key_head_dim": 96,
  "linear_value_head_dim": 192,
  "linear_conv_kernel_dim": 4,
  "linear_allow_neg_eigval": true,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "pad_token_id": 0,
  "rope_parameters": null
}
```

</details>

<details>
<summary><strong>Mamba + Attention MoE Hybrid</strong> — <code>GraniteMoeHybridForCausalLM</code> · Mamba + Attention MoE hybrid · <a href="https://huggingface.co/docs/transformers/en/model_doc/granitemoehybrid">HF Docs</a></summary>

```json
{
  "architectures": [
    "GraniteMoeHybridForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "attention_multiplier": 0.015625,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "embedding_multiplier": 12,
  "hidden_act": "silu",
  "hidden_size": 512,
  "initializer_range": 0.1,
  "intermediate_size": 1024,
  "layer_types": [
    "mamba",
    "mamba",
    "mamba",
    "attention",
    "mamba",
    "mamba",
    "mamba",
    "attention"
  ],
  "logits_scaling": 2,
  "mamba_chunk_size": 256,
  "mamba_conv_bias": true,
  "mamba_d_conv": 4,
  "mamba_d_head": 128,
  "mamba_d_state": 128,
  "mamba_expand": 2,
  "mamba_n_groups": 1,
  "mamba_n_heads": 8,
  "mamba_proj_bias": false,
  "max_position_embeddings": 4096,
  "model_type": "granitemoehybrid",
  "normalization_function": "rmsnorm",
  "num_attention_heads": 8,
  "num_experts_per_tok": 2,
  "num_hidden_layers": 8,
  "num_key_value_heads": 8,
  "num_local_experts": 8,
  "output_router_logits": false,
  "position_embedding_type": "nope",
  "residual_multiplier": 0.22,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000,
  "router_aux_loss_coef": 0.0,
  "shared_intermediate_size": 512,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "vocab_size": 49152
}
```

</details>

<details>
<summary><strong>Linear-Attention MoE Hybrid</strong> — <code>Qwen3NextForCausalLM</code> · Linear-attention MoE hybrid · <a href="https://huggingface.co/docs/transformers/v5.8.0/en/model_doc/qwen3_next">HF Docs</a></summary>

```json
{
  "architectures": [
    "Qwen3NextForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "decoder_sparse_step": 1,
  "full_attention_interval": 4,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 1536,
  "linear_conv_kernel_dim": 4,
  "linear_key_head_dim": 128,
  "linear_num_key_heads": 8,
  "linear_num_value_heads": 8,
  "linear_value_head_dim": 128,
  "max_position_embeddings": 4096,
  "mlp_only_layers": [],
  "model_type": "qwen3_next",
  "moe_intermediate_size": 256,
  "norm_topk_prob": true,
  "num_attention_heads": 8,
  "num_experts": 8,
  "num_experts_per_tok": 2,
  "num_hidden_layers": 8,
  "num_key_value_heads": 8,
  "output_router_logits": false,
  "partial_rotary_factor": 0.25,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000000,
  "router_aux_loss_coef": 0.001,
  "shared_expert_intermediate_size": 256,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 49152
}
```

</details>

