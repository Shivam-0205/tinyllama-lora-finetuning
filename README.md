# TinyLlama LoRA Fine-Tuning Project

## Overview

This project demonstrates parameter-efficient fine-tuning (PEFT) of the TinyLlama 1.1B language model using LoRA adapters on multiple lightweight text datasets from Hugging Face. It was developed using Google Colab with NVIDIA Tesla T4 GPUs and integrates Weights & Biases (W&B) for experiment tracking.

---

## Model

- **Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Parameters:** 1.1 Billion
- **Quantization:** 4-bit via BitsAndBytes
- **Fine-tuning Method:** PEFT + LoRA

---

## Datasets Used

- TinyStories
- WikiText-2
- AG News
- DailyDialog
- SQuAD

> All datasets were trimmed to smaller subsets for compatibility with Colab T4 GPUs.

---

## Technologies Used

- Transformers
- PEFT
- LoRA
- BitsAndBytes
- Hugging Face Datasets
- Weights & Biases
- PyTorch

---

## Training Configuration

| Parameter             | Value             |
|-----------------------|-------------------|
| Batch Size            | 1                 |
| Gradient Accumulation | 4                 |
| Learning Rate         | 2e-4              |
| Epochs                | 1                 |
| Max Length            | 128               |
| Optimizer             | paged_adamw_8bit  |

---

## Performance Metrics

| Metric                | Value     |
|-----------------------|-----------|
| Final Training Loss   | ~2.02     |
| Inference Time        | ~6.1 sec  |
| Tokens/sec            | ~8.86     |

---

## W&B Experiment Tracking

Training runs were tracked using Weights & Biases.

**Logged Metrics:**
- `train/loss`
- `train/learning_rate`
- `train/grad_norm`
- `train/epoch`
- `train/global_step`

---

## Repository Structure

```
## Repository Structure

```text
tinyllama-lora/
├── configs/
│   ├── datasets.yaml
│   ├── lora.yaml
│   └── training.yaml
│
├── data/
│
├── notebooks/
│   └── tinyllama_lora_training.ipynb
│
├── output/
│
├── reports/
│   └── report.tex
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   └── inference.py
│
├── utils/
│   └── logger.py
│
├── README.md
├── documentation.md
└── requirements.txt
```

---

## Running the Project

**Training:**
```bash
python scripts/train.py
```

**Inference:**
```bash
python scripts/inference.py
```

---

## Technical Documentation

### Objective

Fine-tune a small autoregressive language model using parameter-efficient techniques under limited GPU resources.

---

### Model Architecture

TinyLlama 1.1B Chat v1.0 — a lightweight autoregressive transformer model compatible with Hugging Face Transformers and PEFT.

---

### Quantization

The model was loaded using **4-bit quantization** via BitsAndBytes to reduce GPU memory usage.

**Benefits:**
- Lower VRAM usage
- Faster model loading
- T4 GPU compatibility

---

### LoRA Fine-Tuning

LoRA adapters were applied on:
- `q_proj`
- `v_proj`

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Rank (r)  | 8     |
| Alpha     | 16    |
| Dropout   | 0.05  |

---

### Dataset Pipeline

1. Loaded from Hugging Face
2. Cleaned into text-only format
3. Tokenized using TinyLlama tokenizer
4. Truncated to max length 128

---

### Training Pipeline

- Hugging Face `Trainer`
- FP16 mixed precision
- 4-bit quantization
- Gradient accumulation

---

### Inference Testing

Inference tested using Hugging Face `pipeline` API.

**Measured:**
- Latency
- Tokens/sec
- Generated text quality

---

### Challenges Faced

- Colab GPU and runtime limits
- VRAM constraints
- Dataset compatibility issues
- Runtime disconnects

---

## Hardware Used

- Google Colab
- NVIDIA Tesla T4 GPU

---

## Future Improvements

- Multi-epoch fine-tuning
- Validation perplexity evaluation
- Larger dataset integration
- Better response generation quality

---

## Author

**Shivam Naik**
