# Auto-detecting module loader for the Marvin dual-stack (AMD / Intel).
#
# Source this file from any sbatch script:
#
#     source "$workdir/.modules.sh"
#
# Detection rules (first match wins):
#   1. $LLM_FOUNDRY_STACK is set explicitly to "amd" or "intel" -> use it.
#   2. The job requests a GPU via --gres (SLURM_JOB_GRES contains "gpu")   -> AMD.
#   3. The partition name contains "gpu" (case-insensitive)                -> AMD.
#   4. Any other SLURM partition                                           -> Intel.
#   5. No SLURM context                                                    -> Intel (with warning).

_stack=""
_reason=""

if [[ -n "${LLM_FOUNDRY_STACK:-}" ]]; then
    _stack="${LLM_FOUNDRY_STACK,,}"
    _reason="LLM_FOUNDRY_STACK=${LLM_FOUNDRY_STACK}"
elif [[ -n "${SLURM_JOB_GRES:-}" && "${SLURM_JOB_GRES,,}" == *gpu* ]]; then
    _stack="amd"
    _reason="SLURM_JOB_GRES=${SLURM_JOB_GRES} (GPU job -> AMD stack)"
elif [[ -n "${SLURM_JOB_PARTITION:-}" && "${SLURM_JOB_PARTITION,,}" == *gpu* ]]; then
    _stack="amd"
    _reason="SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION} (GPU partition -> AMD stack)"
elif [[ -n "${SLURM_JOB_PARTITION:-}" ]]; then
    _stack="intel"
    _reason="SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION} (CPU partition -> Intel stack)"
else
    _stack="intel"
    _reason="no SLURM context detected; defaulting to Intel stack (set LLM_FOUNDRY_STACK to override)"
    echo "[.modules.sh] WARNING: ${_reason}" >&2
fi

echo "[.modules.sh] Stack: ${_stack}  (${_reason})"

case "${_stack}" in
    amd)
        # AMD-based installations/operations.
        export MODULEPATH=/opt/software/easybuild-AMD/modules/all:/etc/modulefiles:/usr/share/modulefiles:/opt/software/modulefiles:/usr/share/modulefiles/Linux:/usr/share/modulefiles/Core:/usr/share/lmod/lmod/modulefiles/Core
        module purge
        module load CUDA/12.6.0 Python/3.12.3-GCCcore-13.3.0
        ;;
    intel)
        # Intel-based installations/operations.
        module --force purge
        module load Python/3.12.3-GCCcore-13.3.0
        ;;
    *)
        echo "[.modules.sh] ERROR: unknown stack '${_stack}' (expected 'amd' or 'intel')" >&2
        unset _stack _reason
        return 1 2>/dev/null || exit 1
        ;;
esac

# Report what's loaded so the job log shows the resolved environment.
if command -v module >/dev/null 2>&1; then
    echo "[.modules.sh] Loaded modules:"
    module list 2>&1 | sed 's/^/    /'
fi

export LLM_FOUNDRY_STACK="${_stack}"
unset _stack _reason
