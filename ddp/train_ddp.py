"""
Distributed Data Parallel (DDP) Training for Large Language Models

Production-ready training script for transformer-based causal language models using
PyTorch DDP with either standard AdamW or a hybrid Muon + Adam optimizer.
Designed for multi-GPU, multi-node SLURM clusters.

Modules:

- `train_ddp.py`: DDP setup, main training loop, and the resume-from-checkpoint logic.
- `model_setup.py`: Pre-DDP model and tokenizer initialization.
- `data_loading.py`: Dataset loading and DataLoader creation.
- `mfu.py`: MFU calculation utilities for performance monitoring.
- `optimizers.py`: Optimizer and learning rate scheduler creation.
- `specifications.py`: Dataclass definitions for training arguments.
- `utils.py`: Logging, checkpointing, and miscellaneous utilities.

How to Use:

1. Configure `specifications.yaml` with your training settings, including dataset paths, 
    model architecture, and optimization parameters.

2. Launch the training script with SLURM, specifying the number of nodes and GPUs.
    See the `train_ddp.sh` script for an example SLURM job submission.
"""
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch

import contextlib
import argparse
import logging
import time
import yaml
import math
import sys
import os

from codecarbon import EmissionsTracker
import numpy as np
import wandb

from model_setup import prepare_training_components
from specifications import TrainingArguments
from data_loading import prepare_dataloaders

from mfu import (
    calculate_training_metrics, 
    create_mfu_context
)

from optimizers import (
    create_lr_scheduler, 
    create_optimizer, 
    get_optimizer_summary_lines
)

from utils import (
    StructuredTrainingLogger,
    compute_training_schedule,
    setup_triton_cache, 
    cleanup_log_file, 
    checkpoint_already_validated
)

def main(specs, slurm_job_id, hardware):

    # Load the training arguments from the specifications.yaml file
    with open(specs, "r") as stream:
        kwargs = yaml.safe_load(stream)
    
    # Create the `args` object from the loaded specifications.
    # Check the `specifications.py` script to see all available arguments.
    args = TrainingArguments(**kwargs)

    # [Logging facility for Python](https://docs.python.org/3/library/logging.html#)
    logger = logging.getLogger(f"DDP-Trainer-{slurm_job_id}-{args.stage_name}")

    logging.basicConfig(
        format="%(name)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    if "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ:

        # SLURM_NTASKS is the total number of processes (aka, world size).
        world_size = int(os.environ["SLURM_NTASKS"])

        if world_size > 1:

            # SLURM_PROCID is the rank of the current process in SLURM.
            rank = int(os.environ['SLURM_PROCID'])

            # [PyTorch Distributed Documentation](https://docs.pytorch.org/docs/stable/distributed.html)
            dist.init_process_group(
                backend="nccl",
                world_size=world_size, 
                rank=rank,
                device_id=rank % torch.cuda.device_count()
            )

            # Set the device to the current rank.
            device = f"cuda:{rank % torch.cuda.device_count()}"
            torch.cuda.set_device(device)
            # The first process is the master process.
            master_process = rank == 0
            ddp = True
            if master_process:
                logger.info(f"Running DDP via '{dist.get_backend()}' backend. Logging process: {rank}. World size: {world_size}.")
        
        else:
            # If the world size is 1, then we are not using distributed training.
            rank = 0
            device = "cuda:0"
            torch.cuda.set_device(device)
            master_process = True
            ddp = False
            logger.info("Running single process training.")

    else:
        raise ValueError("SLURM_NTASKS or SLURM_PROCID environment variable is not set. This script is intended to be run with SLURM.")

    # Setup Triton cache before any GPU operations.
    setup_triton_cache()

    # If we are `resume_from_checkpoint`, we use the slurm job id from the checkpoint path.
    if args.resume_from_checkpoint:
        slurm_job_id = args.resume_from_checkpoint.split("slurm_job_")[-1].split("/")[0]
    
    # Update the checkpoint directory to include the SLURM job id.
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, f"slurm_job_{slurm_job_id}")
    
    if master_process: 
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        # Create a log file to store the training logs.
        log_file = os.path.join(args.checkpoint_dir, f"{slurm_job_id}.log")
        
        # Clean up the log file if resuming from checkpoint
        if args.resume_from_checkpoint:
            cleanup_log_file(log_file)
        
        file_logger = StructuredTrainingLogger(log_file)

    # [Common PyTorch Functions](https://docs.pytorch.org/docs/stable/torch.html)
    # Set the random state seed for reproducibility.
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # See the `model_setup.py` script for details on what this function does.
    model_state = prepare_training_components(
        args=args,
        device=device,
        master_process=master_process,
        logger=logger,
        file_logger=file_logger if master_process else None,
    )

    # We receive back:
    # - the updated `args` object.
    # - the tokenizer (a HuggingFace tokenizer object).
    # - the model (a PyTorch nn.Module object, still unwrapped in DDP).
    # - the precision (torch.bfloat16 or torch.float32).
    # - the checkpoint path (if resuming from checkpoint, otherwise None).
    # - the number of trainable parameters in the model (int).
    args = model_state.args
    tokenizer = model_state.tokenizer
    model = model_state.model
    precision = model_state.precision
    checkpoint_path = model_state.checkpoint_path
    params = model_state.trainable_params

    if ddp:
        # Wrap the model with DistributedDataParallel (DDP).
        # DDP enables multi-process, multi-GPU training by replicating the model
        # across processes, synchronizing gradients between them, and ensuring
        # efficient scaling across devices. Each process is responsible for one
        # GPU and communicates with others to keep model replicas in sync.
        #
        #        +---------+    +---------+    +---------+
        #        | Rank 0  |    | Rank 1  |    | Rank 2  |
        #        |  GPU 0  |    |  GPU 1  |    |  GPU 2  |
        #        | Model   |    | Model   |    | Model   |
        #        +---------+    +---------+    +---------+
        #            \             |             /
        #             \            |            /
        #              \           |           /
        #               +---------------------+
        #               | Gradient All-Reduce |
        #               +---------------------+
        #
        # Key arguments here:
        #    - device_ids: Pin each process to a specific GPU based on its rank.
        #    - static_graph: 
        #        Enables optimizations for models with a fixed forward/backward
        #        graph (no dynamic control flow). Should be False when using
        #        gradient accumulation or dynamic graphs, since it may otherwise
        #        skip needed synchronizations.
        #    - gradient_as_bucket_view:
        #        Lets gradients be viewed directly from the communication buckets 
        #        to save memory. More efficient, but may complicate custom 
        #        gradient handling.
        # Note:
        #     Gradient accumulation can break if `static_graph=True`
        #     (see https://github.com/pytorch/pytorch/issues/143580).
        #Reference: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        model = DDP(
            model, 
            device_ids=[rank % torch.cuda.device_count()],
            static_graph=args.static_graph,
            gradient_as_bucket_view=True,
        ) 

    # Unwrap version of the model if it is wrapped in DDP.
    raw_model = model.module if ddp else model 

    # See the `data_loading.py` module for details on dataset loading and dataloader creation.
    data = prepare_dataloaders(
        args=args,
        tokenizer=tokenizer,
        world_size=world_size,
        rank=rank,
        logger=logger if master_process else None,
        file_logger=file_logger if master_process else None,
    )

    train_dataloader = data.train_dataloader
    validation_dataloader = data.val_dataloader
    train_sampler = data.train_sampler

    # Calculate gradient accumulation steps, steps per epoch, and total training steps.
    gradient_accumulation_steps, num_update_steps_per_epoch, max_steps = compute_training_schedule(
        args, len(train_dataloader), world_size
    )
    if args.max_steps is not None and master_process:
        logger.info(f"Overriding the number of steps to {max_steps} as per the `max_steps` argument (check the YAML file if you are not sure).")
        file_logger.log_metadata(f"Overriding the number of steps to {max_steps} as per the `max_steps` argument (check the YAML file if you are not sure).")

    lr_scheduler = create_lr_scheduler(args, max_steps)

    if master_process:
        logger.info(f"Using learning rate decay type: {args.lr_decay_type}")
        file_logger.log_metadata(f"Using learning rate decay type: {args.lr_decay_type}")

    # See the `optimizers.py` script for details on what this function does.
    # We receive back: 
    # - the optimizer.
    # - the optimizer step (a function that we will call to update the optimizer).
    # - a label describing the optimizer (for logging purposes).
    optimizer, optimizer_step, optimizer_label = create_optimizer(
        model=model,
        args=args,
        device_type=device_type,
        master_process=master_process,
        logger=logger,
    )

    # If we are resuming from checkpoint, we load the optimizer state from the checkpoint.
    if args.resume_from_checkpoint:

        checkpoint = os.path.join(checkpoint_path, 'checkpoint.pt')
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'), weights_only=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if master_process:
            logger.info(f"Resumed optimizer from checkpoint: {checkpoint_path}")
            file_logger.log_metadata(f"Resumed optimizer from checkpoint: {checkpoint_path}")

    if master_process:

        logger.info("="*50)
        logger.info("***** Running training *****")
        logger.info(f"  Training stage | {args.stage_name}")
        logger.info(f"  SLURM Job ID | {slurm_job_id}")
        logger.info(f"  Hardware | {hardware.upper()}")
        logger.info(f"  World size (total GPUs) | {world_size}")
        logger.info(f"  Precision | {'bfloat16' if args.bf16 else 'float32'}")
        logger.info(f"  Resuming from checkpoint | {args.resume_from_checkpoint is not None}")
        if args.resume_from_checkpoint:
            logger.info(f"    Checkpoint path | {args.resume_from_checkpoint}")
            logger.info(f"    Starting from step | {checkpoint.get('resume_step', None) if not args.begin_new_stage else None}")
        logger.info("="*50)
        logger.info("Dataset Configuration:")
        logger.info(f"  Num train examples | {data.num_train_samples:,}")
        logger.info(f"  Num validation examples | {data.num_val_samples:,}")
        logger.info(f"  Length of train dataloader | {len(train_dataloader):,}")
        logger.info(f"  Max position embeddings (seq length) | {args.max_position_embeddings:,}")
        logger.info(f"  Shuffle dataset | {args.shuffle_dataset}")
        logger.info(f"  Additional mask token IDs | {args.additional_mask_token_ids}")
        logger.info("="*50)
        logger.info("Batch Configuration:")
        logger.info(f"  Num Epochs | {args.num_train_epochs}")
        logger.info(f"  Micro batch size per device | {args.micro_batch_size}")
        logger.info(f"  Gradient accumulation steps | {gradient_accumulation_steps}")
        logger.info(f"  Total batch size (samples) | {args.micro_batch_size * gradient_accumulation_steps * world_size}")
        logger.info(f"  Total batch size (tokens) | {args.total_batch_size:,}")
        logger.info(f"  Total optimization steps | {max_steps:,}")
        logger.info(f"  Steps per epoch | {num_update_steps_per_epoch:,}")
        logger.info(f"  Checkpointing every | {args.checkpointing_steps} steps")
        logger.info("="*50)
        logger.info("Model Architecture:")
        logger.info(f"  Model config | {args.path_to_model_config or args.base_model or 'From checkpoint'}")
        logger.info(f"  Attention implementation | {args.attn_implementation}")
        logger.info(f"  Gradient checkpointing | {args.gradient_checkpointing}")
        logger.info(f"  Liger kernel | {args.use_liger_kernel}")
        logger.info(f"  Torch compile | {args.torch_compile}")
        logger.info(f"  MFU type | {args.mfu_type}")
        logger.info(f"  Trainable parameters | {params:,}")
        logger.info("="*50)
        optimizer_summary_lines = get_optimizer_summary_lines(args)
        logger.info(f"Optimizer Configuration ({optimizer_label}):")
        for line in optimizer_summary_lines:
            logger.info(line)
        logger.info("="*50)

        if args.resume_from_checkpoint is None:
            file_logger.log_metadata("="*50)
            file_logger.log_metadata("***** Running training *****")
            file_logger.log_metadata(f"  Training stage | {args.stage_name}")
            file_logger.log_metadata(f"  SLURM Job ID | {slurm_job_id}")
            file_logger.log_metadata(f"  Hardware | {hardware.upper()}")
            file_logger.log_metadata(f"  World size (total GPUs) | {world_size}")
            file_logger.log_metadata(f"  Precision | {'bfloat16' if args.bf16 else 'float32'}")
            file_logger.log_metadata("="*50)
            file_logger.log_metadata("Dataset Configuration:")
            file_logger.log_metadata(f"  Num train examples | {data.num_train_samples:,}")
            file_logger.log_metadata(f"  Num validation examples | {data.num_val_samples:,}")
            file_logger.log_metadata(f"  Length of train dataloader | {len(train_dataloader):,}")
            file_logger.log_metadata(f"  Max position embeddings (seq length) | {args.max_position_embeddings:,}")
            file_logger.log_metadata(f"  Shuffle dataset | {args.shuffle_dataset}")
            file_logger.log_metadata(f"  Additional mask token IDs | {args.additional_mask_token_ids}")
            file_logger.log_metadata("="*50)
            file_logger.log_metadata("Batch Configuration:")
            file_logger.log_metadata(f"  Num Epochs | {args.num_train_epochs}")
            file_logger.log_metadata(f"  Micro batch size per device | {args.micro_batch_size}")
            file_logger.log_metadata(f"  Gradient accumulation steps | {gradient_accumulation_steps}")
            file_logger.log_metadata(f"  Total batch size (samples) | {args.micro_batch_size * gradient_accumulation_steps * world_size}")
            file_logger.log_metadata(f"  Total batch size (tokens) | {args.total_batch_size:,}")
            file_logger.log_metadata(f"  Total optimization steps | {max_steps:,}")
            file_logger.log_metadata(f"  Steps per epoch | {num_update_steps_per_epoch:,}")
            file_logger.log_metadata(f"  Checkpointing every | {args.checkpointing_steps} steps")
            file_logger.log_metadata("="*50)
            file_logger.log_metadata("Model Architecture:")
            file_logger.log_metadata(f"  Model config | {args.path_to_model_config or args.base_model or 'From checkpoint'}")
            file_logger.log_metadata(f"  Attention implementation | {args.attn_implementation}")
            file_logger.log_metadata(f"  Gradient checkpointing | {args.gradient_checkpointing}")
            file_logger.log_metadata(f"  Liger kernel | {args.use_liger_kernel}")
            file_logger.log_metadata(f"  Torch compile | {args.torch_compile}")
            file_logger.log_metadata(f"  MFU type | {args.mfu_type}")
            file_logger.log_metadata(f"  Trainable parameters | {params:,}")
            file_logger.log_metadata("="*50)
            file_logger.log_metadata(f"Optimizer Configuration ({optimizer_label}):")
            for line in optimizer_summary_lines:
                file_logger.log_metadata(line)
            file_logger.log_metadata("="*50)

    # If we are resuming from a checkpoint (inside the same stage), we need to get the current iteration count, 
    # which is the number of batches processed by the dataloader so far, the current epoch, and the completed steps 
    # from the checkpoint.
    if args.resume_from_checkpoint and not args.begin_new_stage:

        resume_step = int(checkpoint['resume_step'])
        iter_count = int(checkpoint['iteration'])
        epoch = int(checkpoint['epoch'])
        if epoch > 1:
            # Shuffle the sampler every epoch.
            train_sampler.set_epoch(epoch)

    else:
        # For the beginning of training, and every subsequent stage, we reset all counters.
        # WARNING: Don't forget to set `begin_new_stage` to True every time you start a new stage!!!
        resume_step = 0
        iter_count = 0
        epoch = 1

        if args.resume_from_checkpoint and args.begin_new_stage:
            if master_process:
                logger.info(f"Starting new training stage | {args.stage_name}")
                file_logger.log_metadata(f"Starting new training stage | {args.stage_name}")
    
    if not args.begin_new_stage:
        if master_process:
            logger.info(f"WARNING: `begin_new_stage` is set to False. If this is a multistage training, make sure you set it to True for the new stages.")

    # Initialize  W&B and CodeCarbon.
    # If you dont want to use W&B, you maybe should consider checking Trackio.
    # -> https://github.com/gradio-app/trackio
    if master_process:

        if args.wandb_token is not None: 

            # Login to wandb.
            # [wandb.login](https://docs.wandb.ai/ref/python/sdk/functions/login)
            wandb.login(key=args.wandb_token)

            # Initialize wandb.
            # [wandb.init](https://docs.wandb.ai/ref/python/sdk/functions/init/)
            wandb.init(
                project=args.wandb_project if args.wandb_project is not None else "default", 
                notes=args.wandb_desc if args.wandb_desc is not None else "N/A",
                name=f"""{args.wandb_id}-{args.stage_name}-{time.strftime("%d-%m-%Y")}-bs-{args.total_batch_size}-epochs-{args.num_train_epochs}-steps-{max_steps}-lr-{args.max_learning_rate}-sch-{args.lr_decay_type}""",
                config=kwargs,
                resume="allow", # Allows resuming runs that stopped before completion.
                id=f"{args.wandb_id}-{slurm_job_id}" if args.wandb_id is not None else f"{slurm_job_id}",
            )
        
        # We would also like to track the energy consumption of the training process. 
        # For this, we are going to use the `codecarbon` library.
        # To do this, we need to initialize the `EmissionsTracker` and then track 
        # the energy consumption on [only the main process](https://github.com/mlco2/codecarbon/issues/544).
        # [EmissionsTracker](https://mlco2.github.io/codecarbon/usage.html#explicit-object)
        tracker = EmissionsTracker(
            project_name=args.wandb_project if args.wandb_project is not None else "default",
            log_level="critical", # Set to "critical" to silence codecarbon.
            output_dir=args.checkpoint_dir,
            output_file=f"emissions_{slurm_job_id}.csv",
            tracking_mode='machine', # We are tracking the energy consumption of all processes (all GPUS in a given machine/node).
        )

        logger.info(f'Geo Location: ISO: {tracker._geo.country_iso_code} | Country: {tracker._geo.country_name} | Region : {tracker._geo.region}')
        tracker.start()
        # Get the correct MFU calculation settings.
        # See the `mfu.py` script for details on what this function does.
        mfu_context = create_mfu_context(args=args, hardware=hardware, num_parameters=params)

    # Set the model to training mode.
    model.train()

    # Prepare a null context manager to use in combination with `model.no_sync()`
    # This is used to avoid synchronizing gradients during gradient accumulation steps.
    # [nullcontext](https://docs.python.org/3/library/contextlib.html#contextlib.nullcontext)
    null_context = contextlib.nullcontext()

    # Create an iterator from the train dataloader.
    iter_train_dataloader = iter(train_dataloader)

    if master_process:
        logger.info(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
        file_logger.log_metadata(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")

    # Flag to indicate if the learning rate stage has changed (starts as False)
    # We use this to save a checkpoint if the learning rate stage changes.
    lr_stage_change = False
    # Get the current learning rate stage
    current_lr_stage = lr_scheduler(resume_step)[-1]

    # Start the training loop.
    for completed_steps in range(1, max_steps + 1):

        # Skip the steps that have already been completed when resuming from a checkpoint.
        if resume_step >= completed_steps:
            for micro_step in range(gradient_accumulation_steps):
                try:
                    next(iter_train_dataloader)
                except StopIteration:
                    # If we reach the end of the dataloader, we need to reset the iterator.
                    epoch += 1
                    train_sampler.set_epoch(epoch)
                    iter_train_dataloader = iter(train_dataloader)
                    next(iter_train_dataloader)
                iter_count += 1
            continue
            
        # Reset the sampler if we have exhausted the dataloader for this epoch.
        # iter_count tracks the number of BATCHES consumed, not optimizer steps.
        if iter_count >= len(train_dataloader):
            if epoch < math.ceil(args.num_train_epochs):
                epoch += 1
                iter_count = 0
                train_sampler.set_epoch(epoch)
                # IMPORTANT: Must recreate iterator after changing sampler epoch
                iter_train_dataloader = iter(train_dataloader)
                if master_process:
                    logger.info(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
                    file_logger.log_metadata(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
         
        # Evaluate the model when:
        # - We have completed `args.checkpointing_steps` steps (excluding step 0).
        # - The learning rate stage has changed.
        # - We are at the last step.
        if (
            completed_steps % args.checkpointing_steps == 0
            or lr_stage_change
            or completed_steps == max_steps
        ):
            # Check if this checkpoint has already been validated (to avoid re-validation on resume)
            already_validated = checkpoint_already_validated(
                args.checkpoint_dir, 
                args.stage_name, 
                completed_steps, 
                log_file if master_process else os.path.join(args.checkpoint_dir, f"{slurm_job_id}.log")
            )
            
            # Skip validation if checkpoint already exists and has been validated
            if already_validated:
                if master_process:
                    logger.info(f"Skipping validation for step {completed_steps} - checkpoint already validated.")
                pass
            else:
                if master_process:
                    logger.info("***** Running validation *****")

                model.eval()

                with torch.no_grad():

                    val_loss_accum = 0.0
                    num_batches = 0

                    # Time the validation loop.
                    val_t0 = time.time()

                    for _, batch in enumerate(validation_dataloader):

                        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                        
                        with torch.autocast(device_type=device_type, dtype=precision):
                            loss = model(
                                input_ids=batch["input_ids"],
                                labels=batch["labels"],
                            ).loss

                        val_loss_accum += loss.detach()
                        num_batches += 1

                    val_t1 = time.time()
                    val_time = val_t1 - val_t0
                    
                    # Average the loss over the number of batches on this process
                    if num_batches > 0:
                        val_loss_accum = val_loss_accum / num_batches

                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
                    val_loss_accum = val_loss_accum / world_size

                model.train()

                if master_process:
                    logger.info(f"Validation | step: {completed_steps:5d} | loss: {val_loss_accum.item():.4f} | kWh: {tracker._total_energy.kWh:.2f} | val_time: {val_time:.2f}s")
                    file_logger.log_stats(
                        {
                            "status": "validation",
                            "step": completed_steps,
                            "loss": round(val_loss_accum.item(), 6),
                            "kwh": round(tracker._total_energy.kWh, 6),
                            "val_time_s": round(val_time, 6),
                            "stage_name": args.stage_name,
                        }
                    )

                    if args.wandb_token is not None:
                        wandb.log({"val_loss": val_loss_accum.item()})

                    # Create the checkpoint directory.
                    checkpoint_name = f"step_{completed_steps:05d}"
                    output_dir = os.path.join(args.checkpoint_dir, args.stage_name, checkpoint_name)
                    os.makedirs(output_dir, exist_ok=True)

                    # Save the model and tokenizer.
                    raw_model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    # Save the optimizer state and other metadata.
                    torch.save(
                        {
                        'resume_step' : completed_steps,
                        'iteration': iter_count,
                        'epoch': epoch,
                        'config': raw_model.config,
                        'optimizer': optimizer.state_dict(),
                        }, 
                        f"{output_dir}/checkpoint.pt",
                    )
                
                    # Push it to the hub.
                    if args.push_to_hub and args.hub_token is not None and args.hub_model_id is not None:
                        hub_model_id = f"{args.hub_model_id}-{args.stage_name}-{completed_steps}"
                        raw_model.push_to_hub(hub_model_id, token=args.hub_token, private=True)
                        tokenizer.push_to_hub(hub_model_id, token=args.hub_token)

                    # Flush the codecarbon tracker at the end of the validation step.
                    tracker.flush()
                
                # Set barrier to ensure that all processes have finished the validation step before continuing.
                if ddp:
                    dist.barrier()
        
        # We are timing the training loop to measure the MFU.
        t0 = time.time()

        # Initiate a counter for the accumulated loss.
        accumulated_loss = 0.0
        
        # Perform one optimization step.
        optimizer.zero_grad(set_to_none=True)
        
        # Perform the gradient accumulation inner loop.
        for micro_step in range(gradient_accumulation_steps):

            # Get the next batch.
            try:
                batch = next(iter_train_dataloader)
            except StopIteration:
                # Epoch boundary crossed during gradient accumulation
                # Must update sampler epoch before recreating iterator
                epoch += 1
                train_sampler.set_epoch(epoch)
                iter_train_dataloader = iter(train_dataloader)
                batch = next(iter_train_dataloader)
                if master_process:
                    logger.info(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
                    file_logger.log_metadata(f"Epoch {epoch} of {math.ceil(args.num_train_epochs)}")
                # Reset iter_count since we're starting a new epoch
                iter_count = 0

            # Move the batch to the device.
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Use no_sync context manager for all steps except the last one,
            # which is the only one that requires gradient synchronization.
            sync_context = model.no_sync() if ddp and micro_step < gradient_accumulation_steps - 1 else null_context

            with sync_context:
                # Autocast is a PyTorch context manager that enables mixed precision training.
                # [torch.autocast](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast)
                with torch.autocast(device_type=device_type, dtype=precision):

                    # Transformers from HF perform the loss calculation internelly,
                    # as well as the shifting of the labels in the case of a causal language model.
                    # - Note: the loss is already an average loss over the micro-batch.
                    loss = model(
                                input_ids=batch["input_ids"],
                                labels=batch["labels"],
                            ).loss

                # Accumulate the raw loss.
                accumulated_loss += loss.detach()
                
                # Scale the loss for gradient accumulation before backward pass.
                scaled_loss = loss / gradient_accumulation_steps
                scaled_loss.backward()
                # Track batch consumption for epoch boundary detection
                iter_count += 1

        # Average the accumulated loss over gradient accumulation steps.
        accumulated_loss = accumulated_loss / gradient_accumulation_steps

        if ddp:
            dist.all_reduce(accumulated_loss, op=dist.ReduceOp.SUM)
            accumulated_loss = accumulated_loss / world_size

        # Clip gradients up to `args.max_grad_norm`.
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)

        # Determine the learning rate for the current step.
        adam_lr, muon_lr, lr_stage = lr_scheduler(completed_steps)

        # And check if the learning rate stage has changed.
        if lr_stage != current_lr_stage:
            lr_stage_change = True
            current_lr_stage = lr_stage
            if master_process:
                logger.info(f"Learning rate stage changed to: {current_lr_stage} at step {completed_steps} | {args.stage_name}.")
                file_logger.log_metadata(f"Learning rate stage changed to: {current_lr_stage} at step {completed_steps} | {args.stage_name}.")
        else:
            lr_stage_change = False

        optimizer_step(adam_lr, muon_lr, completed_steps)
        torch.cuda.synchronize() if device_type == "cuda" else None
        t1 = time.time()

        if master_process:

            dt = t1 - t0
            # Calculate the MFU and other performance metrics.
            # See the `mfu.py` script for details on what this function does.
            performance_metrics = calculate_training_metrics(
                mfu_context=mfu_context,
                micro_batch_size=args.micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                world_size=world_size,
                dt=dt,
            )
            
            # We receive back:
            # - The global tokens per second (tokens/s processed across all GPUs).
            # - The tokens per second per GPU.
            # - The MFU (the GPU utilization metric). 
            global_tokens_per_sec = performance_metrics.global_tokens_per_sec
            tokens_per_sec_per_gpu = performance_metrics.tokens_per_sec_per_gpu
            mfu = performance_metrics.mfu

            # Get the current VRAM usage.
            if device.startswith("cuda"):
                used_vram = torch.cuda.max_memory_allocated(device) // (1024 ** 3)
            else:
                used_vram = 0

            lr_log = f"adam-lr: {adam_lr:.4e}"
            if muon_lr is not None:
                lr_log += f" | muon-lr: {muon_lr:.4e}"

            logger.info(f"Training | step: {completed_steps:5d} | loss: {accumulated_loss.item():.6f} | {lr_log} | lr stage: '{current_lr_stage}' | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | global tok/sec: {global_tokens_per_sec:.2f} | tok/sec/gpu: {tokens_per_sec_per_gpu:.2f} | VRAM: {used_vram:.2f} | MFU: {mfu:.2f}%")
            training_stats = {
                "status": "training",
                "step": completed_steps,
                "loss": round(accumulated_loss.item(), 6),
                "adam_lr": adam_lr,
                "lr_stage": current_lr_stage,
                "grad_norm": round(float(norm), 6),
                "dt_ms": round(dt * 1000, 6),
                "global_tokens_per_sec": round(global_tokens_per_sec, 6),
                "tokens_per_sec_per_gpu": round(tokens_per_sec_per_gpu, 6),
                "vram_gb": round(float(used_vram), 6),
                "mfu": round(mfu, 6),
                "stage_name": args.stage_name,
            }
            if muon_lr is not None:
                training_stats["muon_lr"] = muon_lr
            file_logger.log_stats(training_stats)

            if args.wandb_token is not None:

                metrics = {
                    "loss": accumulated_loss.item(),
                    "step": completed_steps,
                    "adam_lr": adam_lr,
                    "grad_norm": norm,
                    "dt_ms": dt * 1000,
                    "global_tokens_per_sec": global_tokens_per_sec,
                    "tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
                    "mfu": mfu,
                }
                if muon_lr is not None:
                    metrics["muon_lr"] = muon_lr
                wandb.log(metrics)
    
    # Terminate the W&B tracker and the CodeCarbon tracker at the end of the training loop.
    if master_process:
        tracker.stop()
        if args.wandb_token is not None:
            wandb.finish()

    # Cleanup.
    if ddp:
        dist.destroy_process_group()
    # Done!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Data Parallel Training")
    parser.add_argument(
        "--specs",
        type=str,
        required=True,
        help="The path to the specifications file.",
    )
    parser.add_argument(
        "--slurm-job-id",
        type=str,
        required=True,
        help="The SLURM job id.",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        required=True,
        help="The hardware used for training.",
    )
    args = parser.parse_args()

    main(args.specs, args.slurm_job_id, args.hardware)
