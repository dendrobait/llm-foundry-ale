# Distributed Training

Production-ready distributed training scripts for large language models using PyTorch's DDP (Distributed Data Parallel) and FSDP (Fully Sharded Data Parallel) strategies.

## Overview

This folder contains distributed training scripts for large language models using PyTorch's DDP (Distributed Data Parallel) and FSDP (Fully Sharded Data Parallel) strategies. Both are optimized for multi-GPU, multi-node SLURM clusters and support standard AdamW or hybrid Muon + Adam optimizers. The supported model families are vanilla GPT-style **dense transformers** (e.g. `LlamaForCausalLM`, `Qwen3ForCausalLM`) and the **Qwen3.5** family — dense, MoE, and hybrid linear/full-attention variants (`Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`).

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

Here we have toy examples of model config files covering the supported architectures. Each config is a `transformers`-compatible JSON that can be passed directly to `path_to_model_config` in `specifications.yaml`.

- **NOTE**: For Qwen3.5 hybrids that mix linear-attention layers with full attention (`layer_types` containing `"linear_attention"`), the fast path in the modeling code requires **both** `flash-linear-attention` (gated-delta-rule chunk / fused kernels) **and** `causal-conv1d` (short-conv branch of `GatedDeltaNet`). Install both with `pip install flash-linear-attention causal-conv1d`. If either is missing, training still runs but falls back to a slow PyTorch reference path.

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
<summary><strong>Qwen3.5 Dense (Full Attention)</strong> — <code>Qwen3_5ForCausalLM</code> · Dense transformer with full attention · <a href="https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5">HF Docs</a></summary>

```json
{
  "architectures": [
    "Qwen3_5ForCausalLM"
  ],
  "model_type": "qwen3_5_text",
  "vocab_size": 49152,
  "hidden_size": 512,
  "intermediate_size": 1536,
  "num_hidden_layers": 8,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "head_dim": 64,
  "hidden_act": "silu",
  "max_position_embeddings": 4096,
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-06,
  "use_cache": true,
  "tie_word_embeddings": false,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "partial_rotary_factor": 0.25,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "rope_parameters": null,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "pad_token_id": 0,
  "torch_dtype": "bfloat16"
}
```

</details>

<details>
<summary><strong>Qwen3.5 Hybrid (Linear + Full Attention)</strong> — <code>Qwen3_5ForCausalLM</code> · Dense hybrid · <a href="https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5">HF Docs</a></summary>

Every 4th layer is full attention, the rest use Gated-DeltaNet linear attention.

```json
{
  "architectures": [
    "Qwen3_5ForCausalLM"
  ],
  "model_type": "qwen3_5_text",
  "vocab_size": 49152,
  "hidden_size": 512,
  "intermediate_size": 1536,
  "num_hidden_layers": 8,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "head_dim": 64,
  "hidden_act": "silu",
  "max_position_embeddings": 4096,
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-06,
  "use_cache": true,
  "tie_word_embeddings": false,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "partial_rotary_factor": 0.25,
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
  "linear_num_value_heads": 16,
  "linear_key_head_dim": 64,
  "linear_value_head_dim": 64,
  "linear_conv_kernel_dim": 4,
  "rope_parameters": null,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "pad_token_id": 0,
  "torch_dtype": "bfloat16"
}
```

</details>

<details>
<summary><strong>Qwen3.5 MoE (Full Attention)</strong> — <code>Qwen3_5MoeForCausalLM</code> · MoE transformer with full attention · <a href="https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5_moe">HF Docs</a></summary>

```json
{
  "architectures": [
    "Qwen3_5MoeForCausalLM"
  ],
  "model_type": "qwen3_5_moe_text",
  "vocab_size": 49152,
  "hidden_size": 512,
  "intermediate_size": 1536,
  "num_hidden_layers": 8,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "head_dim": 64,
  "hidden_act": "silu",
  "max_position_embeddings": 4096,
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-06,
  "use_cache": true,
  "tie_word_embeddings": false,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "partial_rotary_factor": 0.25,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "num_experts": 8,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 384,
  "shared_expert_intermediate_size": 384,
  "norm_topk_prob": true,
  "output_router_logits": false,
  "router_aux_loss_coef": 0.001,
  "rope_parameters": null,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "pad_token_id": 0,
  "torch_dtype": "bfloat16"
}
```

</details>

<details>
<summary><strong>Qwen3.5 MoE Hybrid (Linear + Full Attention)</strong> — <code>Qwen3_5MoeForCausalLM</code> · MoE hybrid · <a href="https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5_moe">HF Docs</a></summary>

Every 4th layer is full attention, the rest use Gated-DeltaNet linear attention; MLPs are routed mixture-of-experts with an optional shared expert.

```json
{
  "architectures": [
    "Qwen3_5MoeForCausalLM"
  ],
  "model_type": "qwen3_5_moe_text",
  "vocab_size": 49152,
  "hidden_size": 512,
  "intermediate_size": 1536,
  "num_hidden_layers": 8,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "head_dim": 64,
  "hidden_act": "silu",
  "max_position_embeddings": 4096,
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-06,
  "use_cache": true,
  "tie_word_embeddings": false,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "partial_rotary_factor": 0.25,
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
  "linear_num_value_heads": 16,
  "linear_key_head_dim": 64,
  "linear_value_head_dim": 64,
  "linear_conv_kernel_dim": 4,
  "num_experts": 8,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 384,
  "shared_expert_intermediate_size": 384,
  "norm_topk_prob": true,
  "output_router_logits": false,
  "router_aux_loss_coef": 0.001,
  "rope_parameters": null,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "pad_token_id": 0,
  "torch_dtype": "bfloat16"
}
```

</details>

