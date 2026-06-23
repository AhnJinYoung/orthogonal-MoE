# Orthogonal MoE Aggregation Experiments

This repository contains a runnable scaffold for testing token-wise orthogonal aggregation in sparse Mixture-of-Experts LLMs.

The default target is `google/gemma-4-26B-A4B`; Qwen is configured through `configs/qwen35_35b_a3b.yaml`. The code patches Hugging Face MoE expert modules after loading the checkpoint, so the same checkpoint can be evaluated under several aggregation rules without editing Hugging Face source code.

## What is implemented

Aggregation variants live in `src/orthomoe/aggregations.py`:

1. `standard`: the normal router-weighted top-k sum.
2. `top1_orthogonal`: remove non-top experts' component along the top-1 expert direction.
3. `weighted_sum_top1_projection`: project the weighted non-top aggregate once.
4. `gram_schmidt`: sequentially orthogonalize selected experts in router order.
5. `whitening`: symmetric per-token whitening over selected expert outputs.
6. `novelty_gated`: keep expert outputs unchanged but downweight redundant non-top experts.
7. `norm_matched_shrinkage`: control that only shrinks non-top outputs.
8. `random_anchor_projection`: control that projects against a random selected expert.

The Hugging Face patching logic is in `src/orthomoe/hf_patch.py`. It supports the tensor-expert layout used by Gemma4, Qwen3MoE, and Qwen3.5 MoE style modules:

```text
gate_up_proj: [num_experts, 2 * intermediate_dim, hidden_dim]
down_proj:    [num_experts, hidden_dim, intermediate_dim]
forward(hidden_states, top_k_index, top_k_weights)
```

It also includes a fallback for Mixtral-like `gate + ModuleList(experts)` MoE blocks.

## Repository layout

```text
orthogonal_moe_experiments/
  README.md
  requirements.txt
  configs/
    default_gemma4_26b.yaml
    qwen35_35b_a3b.yaml
    pretrain_gemma4_from_scratch.yaml
    smoke_tiny_moe.yaml
  scripts/
    env.sh                # source on every pod: activate PVC venv + caches
    setup_env.sh          # one-time: create venv on the PVC + install deps
    run_all.sh
    smoke_test.sh
    inspect_model.py
  k8s/
    orthomoe-pod.yaml     # example PVC + pod with CPU/RAM limits
  src/orthomoe/
    aggregations.py
    hf_patch.py
    losses.py
    model_loader.py
    data.py
    benchmark.py
    logit_metrics.py      # memory-safe detailed metrics (PPL, accuracy, entropy, margin)
    resources.py          # CPU/RAM thread caps, max_memory loading, RAM guard
    visualize.py
    generate.py
    train.py
    mini_moe_model.py
  tests/
    test_aggregations.py
```

## Installation on an ephemeral pod with a persistent volume (recommended)

These experiments run on temporary `kubectl` pods that die frequently but share
a persistent volume (PVC). To avoid reconfiguring on every restart, the venv,
the Hugging Face hub/dataset caches, the pip cache and the `device_map` offload
folder all live under `$ORTHOMOE_PVC` (default `/pvc/orthomoe`).

**One-time, after the PVC is first mounted:**

```bash
export ORTHOMOE_PVC=/pvc/orthomoe          # point at your PVC mount
# Optional: CUDA-matched torch wheel index for the install
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
bash scripts/setup_env.sh                  # creates venv on the PVC + installs deps
```

**Every time a new pod starts** (seconds, no downloads/installs):

```bash
export ORTHOMOE_PVC=/pvc/orthomoe
source scripts/env.sh                       # activates the persisted venv + caches
```

`scripts/env.sh` exports `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`,
`PIP_CACHE_DIR`, `TORCH_HOME`, `TRITON_CACHE_DIR` and `ORTHOMOE_OFFLOAD` into the
PVC, activates the venv, and caps CPU threads. `setup_env.sh` is idempotent: it
hashes `requirements.txt` and skips reinstalling when nothing changed
(`FORCE=1` reinstalls). `scripts/run_all.sh` auto-runs `setup_env.sh` on the
first pod if the venv is missing.

If `/pvc` is not mounted, both scripts fall back to a repo-local `.pvc/`
directory so the same workflow runs on a laptop.

A ready-to-edit pod + PVC manifest is in [`k8s/orthomoe-pod.yaml`](k8s/orthomoe-pod.yaml).
Keep the container memory limit `>=` the config's `resources.max_cpu_ram_gb`.

If the checkpoint requires access approval:

```bash
huggingface-cli login                       # token is cached on the PVC
```

### Plain (non-PVC) install

```bash
conda create -n orthomoe python=3.11 -y && conda activate orthomoe
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
pip install -r requirements.txt
# If transformers>=5.12.0 is unavailable:
pip install -U git+https://github.com/huggingface/transformers.git
```

## Run the full inference benchmark

From the repository root:

```bash
bash scripts/run_all.sh configs/default_gemma4_26b.yaml gemma4_first_run
```

This runs:

```text
benchmark -> visualization -> sample generation
```

Outputs are written to:

```text
outputs/gemma4_first_run/
  benchmark/benchmark.jsonl
  benchmark/benchmark.csv
  benchmark/run_config.json
  benchmark/logit_stats/<variant>.npz   # per-token entropy/margin/nll/top1_prob samples
  figures/*.png
  generations.jsonl
```

For Qwen:

```bash
bash scripts/run_all.sh configs/qwen35_35b_a3b.yaml qwen35_first_run
```

## Run only the benchmark

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
python -m orthomoe.benchmark \
  --config configs/default_gemma4_26b.yaml \
  --output outputs/gemma4_benchmark
```

## Visualize an existing run

```bash
python -m orthomoe.visualize \
  --results outputs/gemma4_benchmark/benchmark.jsonl \
  --outdir outputs/gemma4_benchmark/figures
```

The visualization script reads `benchmark.jsonl` and the per-variant
`logit_stats/*.npz` and produces:

```text
# quality / efficiency
perplexity_by_variant.png
bits_per_token_by_variant.png
loss_by_variant.png
throughput_by_variant.png
peak_memory_by_variant.png

# next-token prediction quality
token_accuracy_by_variant.png
top5_accuracy_by_variant.png

# output-logit expressiveness (how the distribution changes)
pred_entropy_by_variant.png         # predictive entropy in bits
effective_classes_by_variant.png    # exp(entropy): how many tokens compete
logit_margin_by_variant.png         # top1 - top2 logit
top1_prob_by_variant.png            # top-1 softmax confidence
dist_entropy.png                    # full per-token entropy distribution overlay
dist_margin.png                     # per-token logit-margin distribution overlay
dist_top1_prob.png                  # per-token top-1 probability distribution
dist_nll.png                        # per-token NLL distribution

# MoE geometry + relationships + summaries
cos_top1_abs_by_variant.png
novelty_by_variant.png
ppl_vs_cos_top1_abs.png
ppl_vs_throughput.png
ppl_vs_entropy.png
accuracy_vs_margin.png
expressiveness_delta_vs_standard.png   # grouped relative change vs the standard baseline
metric_heatmap.png                     # z-scored variant x metric heatmap
```

The `dist_*.png` overlays and `expressiveness_delta_vs_standard.png` are the
direct "how expressive did the model become" views: they show the **full shape**
of the output distribution per variant, not just the mean. A variant that
sharpens predictions shifts entropy down and margin/top-1 confidence up; a more
hedging variant does the opposite.

## Continue pretraining from a pretrained checkpoint

This is the experiment:

> pretrained model + new aggregation during training + same new aggregation during inference

Run:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
python -m orthomoe.train \
  --config configs/default_gemma4_26b.yaml \
  --output outputs/gemma4_continue_pretrain_top1ortho
```

By default, the Gemma config uses LoRA-style continued pretraining. Full AdamW training of a 26B/35B model is not realistic on a single A100 80GB because optimizer state alone can exceed GPU memory. For full-parameter training, use a smaller model, CPU/NVMe offload, or multi-GPU FSDP/ZeRO.

You can also run training as part of the whole pipeline:

```bash
RUN_TRAIN=1 bash scripts/run_all.sh configs/default_gemma4_26b.yaml gemma4_train_run
```

## Pretrain from scratch with the new aggregation

This is the experiment:

> randomly initialized model with new aggregation during pretraining + same new aggregation during inference

The provided config instantiates Gemma4 from config, then patches the MoE experts:

```bash
python -m orthomoe.train \
  --config configs/pretrain_gemma4_from_scratch.yaml \
  --output outputs/gemma4_scratch_top1ortho
```

This mode is included for correctness and cluster-scale runs. A full 26B scratch pretrain is not feasible on one A100 80GB. For a quick code-path test, use the tiny native MoE model:

```bash
bash scripts/smoke_test.sh
```

## Change the model

Edit only the `model.model_id` field in a config:

```yaml
model:
  model_id: Qwen/Qwen3.5-35B-A3B-Base
  dtype: bfloat16
  device_map: auto
```

Then run:

```bash
bash scripts/run_all.sh configs/your_model.yaml your_model_run
```

To debug whether a model exposes compatible MoE modules:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
python scripts/inspect_model.py --model_id Qwen/Qwen3.5-35B-A3B-Base
python scripts/inspect_model.py --model_id Qwen/Qwen3.5-35B-A3B-Base --load
```

Look for modules with `gate_up_proj`, `down_proj`, `act_fn`, and `num_experts`.

## Recommended first experiment matrix

Start with these variants from `configs/default_gemma4_26b.yaml`:

```text
standard
top1_pos_lam025
top1_pos_lam050
top1_pos_lam100
top1_signed_lam050
top1_pos_lam050_renorm
novelty_gated_alpha1
gram_schmidt_lam050
whitening_lam025
shrinkage_control_075
random_anchor_control
```

The controls matter:

- `shrinkage_control_075` checks whether gains come from smaller residual updates rather than diversity.
- `random_anchor_control` checks whether any improvement is just regularization noise.
- `top1_pos_lam050_renorm` separates orthogonality from norm shrinkage.

## Metrics logged

`benchmark.csv` includes, per variant:

```text
# language-model quality (computed from the output logits, memory-safe streaming)
loss                 # mean next-token NLL (nats)
perplexity           # exp(loss)
bits_per_token       # loss / ln(2)
token_accuracy       # next-token top-1 accuracy
top5_accuracy        # next-token top-5 accuracy

# output-logit expressiveness
pred_entropy         # mean softmax entropy (nats)
pred_entropy_bits    # mean softmax entropy (bits)
effective_classes    # exp(entropy): effective vocabulary support
logit_margin         # mean top1 - top2 logit
top1_prob            # mean probability on the argmax token

# efficiency
tokens_per_second
cuda_peak_allocated_gb
cuda_allocated_gb
cuda_reserved_gb

# MoE geometry (from the patched aggregator)
stat_cos_top1_mean
stat_cos_top1_abs_mean
stat_cos_top1_pos_mean
stat_novelty_top1_mean
stat_gate_entropy
stat_gate_top1_mass
stat_projection_norm_ratio
```

The detailed logit metrics are computed in chunks over the token dimension
(`eval.logit_chunk_size`) so a full `[tokens, vocab]` softmax is never
materialised at once. Disable them with `eval.compute_logit_metrics: false`.

Interpretation:

- Lower `stat_cos_top1_abs_mean` means selected expert outputs are less redundant.
- Higher `stat_novelty_top1_mean` means non-top experts contribute more orthogonal directions.
- If perplexity improves but `stat_projection_norm_ratio` is much lower, compare against the shrinkage control.
- If `random_anchor_control` performs similarly to top-1 projection, the benefit may be generic regularization rather than top-1 geometry.

## Implementation notes

The patch computes raw selected expert outputs as `[tokens, top_k, hidden_dim]`, applies one aggregation variant, then returns the usual `[tokens, hidden_dim]` MoE result to the original model. The router, selected expert indices, expert weights, residual connections, layer norms, and attention blocks remain unchanged.

For Gemma4-style decoder layers, the original flow is approximately:

```python
_, top_k_weights, top_k_index = router(hidden_states_flat)
hidden_states_2 = experts(hidden_states_2, top_k_index, top_k_weights)
```

The patch replaces only `experts.forward(...)`.

## Safety checks before expensive runs

Run unit tests:

```bash
PYTHONPATH=$PWD/src pytest tests/test_aggregations.py
```

Run the tiny smoke test:

```bash
bash scripts/smoke_test.sh
```

Then inspect the target model:

```bash
python scripts/inspect_model.py --model_id google/gemma-4-26B-A4B --load
```

Finally run a short benchmark by setting in the config:

```yaml
data:
  max_eval_samples: 32
eval:
  max_batches: 8
```

## Avoiding exit code 137 (CPU/RAM OOM kills)

Exit code 137 is a SIGKILL from the kernel OOM-killer, almost always host RAM
(not VRAM) running out — most often **while loading** the 26B/35B checkpoint or
while `datasets` tokenizes. Every config now has a `resources:` block that bounds
the CPU/RAM workload explicitly:

```yaml
resources:
  num_threads: 4          # cap BLAS/OpenMP/tokenizer threads (avoids thread-fanout RAM)
  max_cpu_ram_gb: 48      # host RAM budget; null -> auto-detect cgroup limit * headroom
  warn_fraction: 0.85     # log a warning when RSS crosses this fraction of the budget
  abort_fraction: 0.95    # raise a clean MemoryError before the kernel SIGKILLs the pod
  cpu_weight_fraction: 0.7 # share of the budget usable for weights during loading
  max_gpu_mem_gb: null    # per-GPU cap for device_map (null -> accelerate decides)
  offload_folder: null    # null -> $ORTHOMOE_OFFLOAD on the PVC; spills weights to disk
```

What this does:

1. **Bounded loading.** When a RAM budget is set, `from_pretrained` is given a
   `max_memory` map plus a disk `offload_folder`, so a checkpoint larger than
   host RAM streams to disk instead of OOM-killing the pod during load.
2. **Thread caps.** Pods usually advertise every node core; uncapped BLAS/OpenMP/
   tokenizer pools each size to that and multiply the resident footprint.
   `num_threads` (and `scripts/env.sh`) cap them.
3. **A RAM guard.** `MemoryGuard.check()` runs inside the eval/train loops. When
   RSS crosses `abort_fraction` of the budget it frees caches and, if still over,
   raises `MemoryError` with a readable traceback instead of a silent exit 137.
4. **Memory-safe data.** Defaults are `num_workers: 0` (no forked workers that
   duplicate RAM), `map_num_proc: 1`, and small `writer_batch_size`. The detailed
   logit metrics stream in chunks so the vocab-sized softmax never lands in RAM
   all at once.

If `bash scripts/run_all.sh configs/default_gemma4_26b.yaml gemma4_first_run`
still gets killed, in order: lower `resources.max_cpu_ram_gb` to match the pod's
actual limit, set `max_gpu_mem_gb`/`offload_folder` to force disk offload, reduce
`data.max_eval_samples` / `eval.max_batches` / `data.block_size`, or set
`model.quantization: {type: 4bit, ...}`. Match the pod's
`resources.limits.memory` to `max_cpu_ram_gb` (see `k8s/orthomoe-pod.yaml`).

## Main caveat

Patching forward aggregation is intentionally invasive. If a future Hugging Face model changes the expert module layout, `patch_model(...)` may fail or patch zero modules. Use `scripts/inspect_model.py --load` and either adjust `module_name_regex` or add a model-specific patch in `hf_patch.py`.
