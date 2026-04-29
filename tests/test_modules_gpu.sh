#!/bin/bash -l

#############################################
# DUMMY TEST SCRIPT (GPU / AMD stack)
#############################################
# This script is NOT meant to be submitted with sbatch -- it just runs
# locally and FAKES the SLURM environment variables that .modules.sh
# inspects, so we can verify the auto-detection picks the AMD stack.
#
# Run it with:
#     bash tests/test_modules_gpu.sh
#############################################

#SBATCH --account=ag_bit_flek
#SBATCH --partition=sgpu_long
#SBATCH --job-name=test-modules-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:a100:4
#SBATCH --exclusive

#############################################
# Fake the SLURM environment that sbatch would normally inject.
# These mirror the #SBATCH directives above.
#############################################
export SLURM_JOB_PARTITION="sgpu_long"
export SLURM_JOB_GRES="gpu:a100:4"

# Make sure no previous run leaked an override into our shell.
unset LLM_FOUNDRY_STACK

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "================================================================"
echo " DUMMY TEST: GPU partition + --gres=gpu  -->  expect AMD stack"
echo "================================================================"
echo "SLURM_JOB_PARTITION = ${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_GRES      = ${SLURM_JOB_GRES}"
echo "----------------------------------------------------------------"

source "$repo_root/.modules.sh"

echo "----------------------------------------------------------------"
echo "Resolved LLM_FOUNDRY_STACK = ${LLM_FOUNDRY_STACK}"
if [[ "${LLM_FOUNDRY_STACK}" == "amd" ]]; then
    echo "RESULT: PASS"
else
    echo "RESULT: FAIL (expected 'amd', got '${LLM_FOUNDRY_STACK}')"
fi
