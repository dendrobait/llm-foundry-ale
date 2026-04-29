#!/bin/bash -l

#############################################
# DUMMY TEST SCRIPT (CPU / Intel stack)
#############################################
# This script is NOT meant to be submitted with sbatch -- it just runs
# locally and FAKES the SLURM environment variables that .modules.sh
# inspects, so we can verify the auto-detection picks the Intel stack.
#
# Run it with:
#     bash tests/test_modules_cpu.sh
#############################################

#SBATCH --account=ag_bit_flek
#SBATCH --partition=lm_medium
#SBATCH --job-name=test-modules-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:05:00
#SBATCH --mem=1900G
#SBATCH --exclusive

#############################################
# Fake the SLURM environment that sbatch would normally inject.
# These mirror the #SBATCH directives above. Note: no --gres, so
# SLURM_JOB_GRES is left unset (CPU-only job).
#############################################
export SLURM_JOB_PARTITION="lm_medium"
unset SLURM_JOB_GRES

# Make sure no previous run leaked an override into our shell.
unset LLM_FOUNDRY_STACK

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "================================================================"
echo " DUMMY TEST: CPU partition, no --gres  -->  expect Intel stack"
echo "================================================================"
echo "SLURM_JOB_PARTITION = ${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_GRES      = ${SLURM_JOB_GRES:-<unset>}"
echo "----------------------------------------------------------------"

source "$repo_root/.modules.sh"

echo "----------------------------------------------------------------"
echo "Resolved LLM_FOUNDRY_STACK = ${LLM_FOUNDRY_STACK}"
if [[ "${LLM_FOUNDRY_STACK}" == "intel" ]]; then
    echo "RESULT: PASS"
else
    echo "RESULT: FAIL (expected 'intel', got '${LLM_FOUNDRY_STACK}')"
fi
