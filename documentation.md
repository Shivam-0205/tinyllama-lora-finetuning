# Technical Documentation

## Project Objective

The objective of this project was to fine-tune a small autoregressive language model using parameter-efficient fine-tuning techniques under limited GPU resources.

---

## Selected Model

Model:
- TinyLlama/TinyLlama-1.1B-Chat-v1.0

Reasons for selection:
- Lightweight (1.1B parameters)
- Hugging Face compatible
- PEFT + LoRA support
- Suitable for Colab T4 GPUs

---

## Quantization Strategy

The model was loaded using 4-bit quantization via BitsAndBytes.

Advantages:
- Reduced VRAM usage
- Faster loading
- Feasible training on T4 GPUs

---

## LoRA Configuration

LoRA adapters were applied on:
- q_proj
- v_proj

Hyperparameters:

| Parameter | Value |
|---|---|
| r | 8 |
| alpha | 16 |
| dropout | 0.05 |

---

## Dataset Pipeline

Datasets used:
- TinyStories
- WikiText-2
- AG News
- DailyDialog
- SQuAD

Pipeline:
1. Dataset loading from Hugging Face
2. Text cleaning
3. Tokenization
4. Truncation to sequence length 128
5. Concatenation into combined dataset

---

## Training Pipeline

Training stack:
- Transformers Trainer
- PEFT
- BitsAndBytes
- FP16 mixed precision

Training settings:
- Batch size: 1
- Gradient accumulation: 4
- Learning rate: 2e-4
- Epochs: 1

---

## Experiment Tracking

Weights & Biases (W&B) was used to track:
- training loss
- learning rate
- gradient norm
- epoch progression
- runtime statistics

---

## Inference Evaluation

Metrics measured:
- inference latency
- tokens per second
- generated response quality

Observed metrics:
- inference time: ~6.1 sec
- tokens/sec: ~8.86

---

## Challenges Encountered

- Colab GPU usage limits
- Runtime disconnections
- Limited VRAM
- Dataset compatibility issues

---

## Future Improvements

- Train on larger datasets
- Increase epochs
- Add validation perplexity evaluation
- Improve generation quality
