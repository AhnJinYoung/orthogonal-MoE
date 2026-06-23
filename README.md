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
    run_all.sh
    smoke_test.sh
    inspect_model.py
  src/orthomoe/
    aggregations.py
    hf_patch.py
    losses.py
    model_loader.py
    data.py
    benchmark.py
    visualize.py
    generate.py
    train.py
    mini_moe_model.py
  tests/
    test_aggregations.py
```

## Installation

Create a fresh environment on the A100 server:

```bash
conda create -n orthomoe python=3.11 -y
conda activate orthomoe

# Install PyTorch for your CUDA stack. Example only:
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

pip install -r requirements.txt
```

If `transformers>=5.12.0` is not available in your environment, install a source build:

```bash
pip install -U git+https://github.com/huggingface/transformers.git
```

Set your Hugging Face token if the checkpoint requires access approval:

```bash
huggingface-cli login
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

The visualization script produces:

```text
perplexity_by_variant.png
loss_by_variant.png
throughput_by_variant.png
cos_top1_abs_by_variant.png
novelty_by_variant.png
peak_memory_by_variant.png
ppl_vs_cos_top1_abs.png
ppl_vs_throughput.png
```

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

`benchmark.csv` includes:

```text
loss
perplexity
tokens_per_second
cuda_peak_allocated_gb
stat_cos_top1_mean
stat_cos_top1_abs_mean
stat_cos_top1_pos_mean
stat_novelty_top1_mean
stat_gate_entropy
stat_gate_top1_mass
stat_projection_norm_ratio
```

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

## Main caveat

Patching forward aggregation is intentionally invasive. If a future Hugging Face model changes the expert module layout, `patch_model(...)` may fail or patch zero modules. Use `scripts/inspect_model.py --load` and either adjust `module_name_regex` or add a model-specific patch in `hf_patch.py`.
