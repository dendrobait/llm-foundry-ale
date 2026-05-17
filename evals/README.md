# Evaluation

Evaluation scripts for running language model evaluations on multiple language benchmarks using the Language Model Evaluation Harness.

## Contents

- [`eval_harness_bn.sh`](eval_harness_bn.sh) — SLURM job submission script for evaluating models on Bengali language benchmarks.
- [`eval_harness_hi.sh`](eval_harness_hi.sh) — SLURM job submission script for evaluating models on Hindi language benchmarks.
- [`eval_harness_pt.sh`](eval_harness_pt.sh) — SLURM job submission script for evaluating models on Portuguese language benchmarks.
- [`eval_harness_pt_old.sh`](eval_harness_pt_old.sh) — SLURM job submission script for evaluating models on Garcia's Portuguese evaluation harness.

## Usage Summary

### `eval_harness_{lang}.sh`

Submits a SLURM job to evaluate a model on language benchmarks using the Language Model Evaluation Harness.

Example:
```bash
sbatch eval_harness_bn.sh
```

Configure the following in the script before submission:
- `--account` — Your SLURM account
- `--partition` — Target GPU partition
- `--nodes` — Number of compute nodes
- `--ntasks-per-node` — Number of GPUs per node
- `model_name_or_path` — Path or HuggingFace hub ID of the model to evaluate
- `tasks` — Comma-separated list of benchmark tasks to run (Bengali benchmarks)
- `output_dir` — Directory to save evaluation results
- `num_fewshot` — Number of few-shot examples (typically 0, 5, or 10)

## Notes

- The evaluation harness uses EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework to benchmark model performance.
- Results are saved as JSON files in the specified `output_dir` for further analysis.
- Ensure the model checkpoint or HuggingFace hub ID is accessible before submission.
- GPU memory and runtime requirements depend on model size and task complexity; adjust `--mem` and `--time` accordingly.
