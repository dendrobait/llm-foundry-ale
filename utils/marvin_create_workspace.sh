#!/bin/bash -l

#############################################
# Workspace Setup Script for Marvin HPC Cluster
#############################################
# This script allocates a workspace on the Marvin HPC cluster,
# clones the repository, and prepares the directory structure.
#
# Dual stack documentation:
# https://wiki.hpc.uni-bonn.de/en/dualstacks
#############################################

# ----------- User Customization Section -----------
username="nklugeco_hpc"        # <-- Change to your username
file_system="mlnvme"           # <-- Change to your filesystem
work_group="ag_bit_flek"       # <-- Change to your work group
email="kluge@uni-bonn.de"      # <-- Change to your email
remainder=7                    # <-- Change as needed
num_days=90                    # <-- Change as needed
workspace_name="polyglot"      # <-- Change to your workspace/project name

# Workspace directory (constructed from the above variables)
workdir="/lustre/$file_system/data/$username-$workspace_name"

#############################################
# Allocate Workspace
# User Guide to HPC Workspaces:
# https://github.com/holgerBerger/hpc-workspace/blob/1.4.0/user-guide.md
#############################################
ws_allocate -F $file_system -G $work_group -m $email -r $remainder -d $num_days -n $workspace_name
echo "Workspace allocated!"
echo "Username: $username"
echo "File System: $file_system"
echo "Work Group: $work_group"
echo "Workspace Name: $workspace_name"
echo "Workspace Directory: $workdir"

#############################################
# Clone Repository
#############################################
cd $workdir
git clone --branch main https://github.com/Polygl0t/llm-foundry.git
echo "Workspace ready."

#############################################
# Stack Sourcing (.modules.sh)
#############################################
# Marvin has a dual software stack (AMD and Intel). Instead of two
# separate module files, this repo ships a single auto-detecting
# loader at the repository root: `.modules.sh`.
#
# How it picks a stack (first match wins):
#
#   1. $LLM_FOUNDRY_STACK is set to "amd" or "intel"   -> use it
#   2. SLURM_JOB_GRES contains "gpu"  (--gres=gpu:...) -> AMD
#   3. SLURM_JOB_PARTITION contains "gpu"              -> AMD
#   4. Any other SLURM partition                       -> Intel
#   5. No SLURM context (e.g. login node)              -> Intel + warning
#
# Inside a SLURM job, sbatch sets SLURM_JOB_GRES / SLURM_JOB_PARTITION
# automatically from your #SBATCH directives, so the right stack is
# selected without any extra configuration:
#
#   source $workdir/.modules.sh
#
# On a login node (no SLURM context) you must force the stack when
# creating venvs or running interactive commands:
#
#   LLM_FOUNDRY_STACK=amd   source $workdir/.modules.sh
#   LLM_FOUNDRY_STACK=intel source $workdir/.modules.sh
#
# When sourced, the script logs the chosen stack, the reason, and a
# `module list` so the resolved environment is visible in your job log.

#############################################
# Installing Dependencies
#############################################
# Dependencies are managed via pyproject.toml with optional groups:
#
#   data         - Data processing (datatrove, spacy, stanza, etc.)
#   distributed  - Distributed training (torch, accelerate, flash_attn, etc.)
#   synth        - Synthetic data generation (torch, vllm, etc.)
#   trl          - Fine-tuning with TRL (trl, vllm, flash_attn, etc.)
#
# Each config must be installed on the matching node type so that
# hardware-specific packages (CUDA wheels, flash-attn, etc.) resolve
# correctly. The `.modules.sh` loader handles stack selection for you;
# you just need to submit the install job to the correct partition.
#
#   Config        Stack       SLURM partition (recommended)
#   ------        -----       ---------------
#   data          intel       lm_short (Intel nodes)
#   distributed   amd         mlgpu_short (AMD/GPU nodes)
#   synth         amd         mlgpu_short (AMD/GPU nodes)
#   trl           amd         mlgpu_short (AMD/GPU nodes)
#
# --- Step 1: Create a virtual environment (on the login node) ---
#
# On the login node there is no SLURM context, so force the stack
# explicitly via LLM_FOUNDRY_STACK before creating the venv:
#
#   LLM_FOUNDRY_STACK=amd source $workdir/.modules.sh   # or =intel for "data"
#   python3 -m venv $workdir/.venv_<config>
#   source $workdir/.venv_<config>/bin/activate
#   pip install --upgrade pip
#   deactivate
#   module purge
#
# --- Step 2: Install packages (as a SLURM job on the correct node) ---
#
# Inside the job, sbatch has set SLURM_JOB_GRES / SLURM_JOB_PARTITION
# from your #SBATCH directives, so `.modules.sh` resolves the stack
# automatically -- no LLM_FOUNDRY_STACK override needed.
#
#   sbatch --export=ALL <<'EOF'
#   #!/bin/bash -l
#   #SBATCH --account=<your-account>
#   #SBATCH --partition=<partition>        # see table above
#   #SBATCH --job-name=install-<config>
#   #SBATCH --output=$workdir/run_outputs/install-<config>-%j.out
#   #SBATCH --time=01:00:00
#   #SBATCH --nodes=1
#   #SBATCH --ntasks-per-node=1
#   #SBATCH --threads-per-core=1
#   #SBATCH --mem=500G
#   #SBATCH --gres=gpu:a40:1               # We only need a GPU for the "distributed", "synth", and "trl" configs to resolve hardware-specific packages. Omit for "data".
#   #SBATCH --oversubscribe
#
#   source $workdir/.modules.sh        # auto-detects the stack inside the job
#   source $workdir/.venv_<config>/bin/activate
#   pip install -e /path/to/llm-foundry/.[<config>]
#   EOF
#
# Example — installing the "distributed" config:
#
#   LLM_FOUNDRY_STACK=amd source $workdir/.modules.sh
#   python3 -m venv $workdir/.venv_distributed
#   source $workdir/.venv_distributed/bin/activate
#   pip install --upgrade pip
#   deactivate
#   module purge
#
#   sbatch --export=ALL <<'EOF'
#   #!/bin/bash -l
#   #SBATCH --account=ag_cst_gabriel
#   #SBATCH --partition=mlgpu_short
#   #SBATCH --job-name=install-distributed
#   #SBATCH --output=/lustre/mlnvme/data/nklugeco_hpc-polyglot/run_outputs/install-distributed-%j.out
#   #SBATCH --time=01:00:00
#   #SBATCH --nodes=1
#   #SBATCH --ntasks-per-node=1
#   #SBATCH --threads-per-core=1
#   #SBATCH --mem=500G
#   #SBATCH --gres=gpu:a40:1
#   #SBATCH --oversubscribe
#
#   source /lustre/mlnvme/data/nklugeco_hpc-polyglot/.modules.sh
#   source /lustre/mlnvme/data/nklugeco_hpc-polyglot/.venv_distributed/bin/activate
#   pip install -e /lustre/mlnvme/data/nklugeco_hpc-polyglot/llm-foundry/.[distributed]
#   pip install flash_attn==2.8.2 --no-build-isolation --no-cache-dir
#   EOF
#############################################