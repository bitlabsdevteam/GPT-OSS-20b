# GPT-OSS-20B Training Logic Tutorial

This tutorial explains how the current `GPT-OSS-20b` training path works and why the major GPU-programming decisions matter.

---

## 1. Big picture

The training stack in `GPT-OSS-20b` is trying to solve one core problem:

> How do we train a GPT-style decoder-only transformer efficiently and correctly on GPU hardware?

At a high level, the training loop does this:

1. load config
2. initialize distributed runtime if needed
3. select device/GPU rank
4. build tokenizer + dataset (when text mode is enabled)
5. build the GPT model
6. run forward pass
7. compute next-token prediction loss
8. run backward pass
9. update weights
10. periodically save checkpoints

That sounds simple, but almost every one of those steps has GPU-performance consequences.

---

## 2. Why decoder-only GPT training works

A decoder-only GPT model is trained using **next-token prediction**.

Example:

```text
Input:  "The GPU is"
Target: "he GPU is f"
```

In practice, the model sees a sequence of token IDs and learns to predict the next token at every position.

Why this is powerful:
- one training example teaches many predictions at once
- it scales naturally to large corpora
- the same architecture can be used later for inference/generation

---

## 3. Tokenization and dataset logic

The current text-mode path uses a **character-level tokenizer**.

Why use it now:
- very easy to understand
- deterministic
- minimal moving parts
- good for proving training/inference correctness end-to-end

Tradeoff:
- character tokenization is not efficient for real production LLMs
- vocabulary is tiny, but sequences become longer
- later we should move to a production tokenizer (BPE / SentencePiece / Hugging Face tokenizer stack)

Current training data flow:

1. read raw text
2. build vocabulary from observed characters
3. encode text into integer token IDs
4. slice text into overlapping windows of length `max_seq_len`
5. create `(x, y)` pairs where `y` is `x` shifted by one token

This is the foundation for autoregressive language modeling.

---

## 4. Model logic

The model has these main parts:

- token embedding
- positional embedding
- stacked transformer blocks
- final layer norm
- output projection to vocabulary logits

### Token embedding
Turns token IDs into vectors.

Why it matters:
- neural networks cannot operate directly on discrete token IDs
- embeddings create dense continuous representations the transformer can reason over

### Positional embedding
Adds information about token order.

Why it matters:
- self-attention alone does not know position
- language depends strongly on order

### Transformer blocks
Each block combines:
- layer norm
- causal self-attention
- feed-forward network (MLP)
- residual connections

Why residual connections matter:
- stabilize deep training
- improve gradient flow
- make larger stacks practical

---

## 5. Why causal masking is required

During training, the model must not "cheat" by seeing future tokens.

So we apply a **causal mask**.

That means token position `t` can only attend to:
- itself
- earlier positions
- never future positions

Why this matters:
- preserves the autoregressive objective
- ensures training matches inference behavior
- without it, loss would be artificially easy and generation would be invalid

---

## 6. Loss function: why cross-entropy

The model outputs logits over the vocabulary for every token position.

Cross-entropy compares:
- predicted token distribution
- actual next token

Why cross-entropy is used:
- standard objective for discrete classification
- directly optimizes next-token prediction quality
- mathematically aligned with language modeling likelihood

---

## 7. Why GPUs matter here

Transformers are extremely matrix-heavy.

Core operations include:
- embedding lookups
- matrix multiplies
- attention score computation
- softmax
- feed-forward projections

GPUs are good at exactly this pattern because they provide:
- massive parallel floating-point throughput
- high memory bandwidth
- efficient execution of large tensor operations

This is why model training moves from toy CPU code to GPU-aware engineering very quickly.

---

## 8. Why mixed precision matters

Mixed precision means not every operation runs in full FP32.
Instead, we use lower-precision formats like:
- FP16
- BF16

### Why use mixed precision?
Because modern GPUs are much faster and more memory-efficient with lower-precision arithmetic.

### Main benefits

#### 1) Higher throughput
Lower-precision math often runs faster on modern GPUs, especially Tensor Core hardware.

#### 2) Lower memory usage
Half-precision tensors use less memory than FP32.
That means:
- larger batch sizes
- longer sequence lengths
- larger models
- fewer out-of-memory failures

#### 3) Better hardware utilization
On A100-class GPUs, BF16/FP16 paths are often the performance sweet spot.

### Why BF16 is especially attractive
BF16 keeps a wider exponent range than FP16.
That means:
- better numerical stability
- lower risk of overflow/underflow
- easier training compared with pure FP16

### Tradeoffs and risks
- some ops still need FP32 for stability
- reductions / normalization / optimizer states may need care
- poor mixed-precision handling can cause NaNs or unstable gradients

### Why autocast is used
`torch.autocast(...)` automatically chooses lower precision for many safe ops while preserving higher precision where needed.

This gives a practical balance:
- faster execution
- less memory pressure
- reduced manual precision management

---

## 9. Why DDP / distributed training matters

Once a model or dataset gets bigger, one GPU is rarely enough.

Distributed Data Parallel (DDP) works by:
1. copying the model to multiple GPUs
2. splitting different mini-batches across them
3. computing gradients independently on each GPU
4. synchronizing gradients before optimizer update

### Benefits of DDP
- faster training via parallel throughput
- better hardware utilization
- scalable baseline before more advanced parallel methods

### Tradeoff
DDP increases communication cost, especially gradient synchronization.

If communication becomes expensive relative to compute, scaling efficiency drops.

---

## 10. Why checkpointing matters

Checkpoints save training state so work is not lost.

Current checkpoint logic stores:
- model weights
- optimizer state
- training step
- model config metadata

### Benefits
- resume training after interruption
- compare model states at different points
- support later inference
- improve reproducibility/debugging

Without checkpointing, long-running GPU jobs are fragile.

---

## 11. Why activation checkpointing matters next

Activation checkpointing trades compute for memory.

Normally, backprop stores many intermediate activations.
This can consume enormous GPU memory.

Activation checkpointing instead:
- stores fewer activations during forward
- recomputes them during backward when needed

### Benefit
- major memory savings
- enables larger models or longer contexts

### Cost
- extra compute time

This trade is often worth it when memory is the bottleneck.

---

## 12. Why synthetic data is still useful

Synthetic token batches are not "real training," but they are still valuable.

Why:
- isolate system performance from data pipeline complexity
- test GPU throughput
- validate distributed logic
- debug memory and correctness paths

So synthetic data is useful for:
- smoke tests
- perf checks
- hardware bring-up

But it is not enough for real model learning.

---

## 13. What advanced GPU programming should mean in this repo

If we say tomorrow’s work should become more advanced, that means going beyond just “it runs.”

It should include topics like:
- bf16/FP16 behavior on A100
- activation checkpointing
- tensor parallelism
- sequence/context parallelism
- communication overhead analysis
- memory fragmentation and allocator behavior
- profiling kernels and step-time bottlenecks
- checkpoint layout for distributed training
- inference-time KV-cache optimization

That is the difference between:
- model code
and
- GPU systems engineering

---

## 14. Recommended next upgrades

To make this training stack meaningfully more advanced:

1. add activation checkpointing in transformer blocks
2. add deterministic synthetic dataloader for perf baselines
3. add resume-from-checkpoint path
4. add validation loop
5. measure tokens/sec, step time, and memory systematically
6. introduce tensor parallel experiments
7. add Hugging Face tokenizer/model compatibility where useful

---

## 15. Summary

The current training logic is doing the right *kind* of work:
- build model
- train on next-token prediction
- use GPUs efficiently
- checkpoint progress
- prepare for distributed scaling

The most important thing to understand is this:

> GPU programming for LLMs is not only about writing model code. It is about balancing compute, memory, communication, numerical stability, and developer ergonomics.

That is why things like mixed precision, checkpointing, distributed training, and profiling are not optional details. They are the core engineering logic.
