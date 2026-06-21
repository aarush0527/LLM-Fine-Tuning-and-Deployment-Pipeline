# LLM Fine-Tuning & Deployment Pipeline

An end-to-end demonstration of the modern LLM lifecycle: **data generation, preprocessing, LoRA-based fine-tuning, evaluation, deployment, safety guardrails, and automated testing**.

This project showcases how a pretrained language model can be adapted, evaluated, served through a production-style API, and protected with lightweight safety mechanisms.

---

## Overview

Large Language Models are more than training scripts. A real-world workflow involves multiple stages:

1. Data collection and preprocessing
2. Dataset validation and cleaning
3. Parameter-efficient fine-tuning
4. Model evaluation
5. Deployment through an API
6. Safety guardrails
7. Automated testing

This repository implements the complete pipeline using a lightweight GPT-style model and demonstrates the engineering components required to move from raw data to an accessible inference service.

---

## Features

### Data Pipeline

* Synthetic dataset generation with realistic noise injection
* Duplicate record simulation
* HTML contamination
* Unicode corruption
* PII-containing examples
* Boilerplate/legal text
* Overly long records

### Data Cleaning

* Unicode normalization
* Duplicate removal
* HTML detection and filtering
* PII detection using regex-based rules
* Boilerplate identification
* Length-based quality filtering
* Deterministic train/validation/test splitting
* Dataset manifest generation

### Fine-Tuning

* LoRA (Low-Rank Adaptation) integration
* Hugging Face Transformers
* GPT-style causal language modeling
* Configurable training parameters
* Checkpointing and logging
* Reproducible training via seeded execution

### Evaluation

* Perplexity
* ROUGE-1
* ROUGE-2
* ROUGE-L
* Token-level F1
* JSON result export

### Model Serving

* FastAPI-based inference service
* Request/response schema validation
* Structured logging
* Rate limiting
* Health monitoring endpoint
* Configurable generation parameters

### Safety Guardrails

* Prompt injection detection
* Jailbreak pattern filtering
* Hallucination trap detection
* Harmful content filtering
* PII leakage detection
* Standardized refusal behavior

### Testing

* Unit tests for preprocessing rules
* Integration tests for cleaning pipeline
* API contract tests
* Guardrail validation tests
* Deterministic pipeline verification

---

## Project Structure

```text
.
├── data
│   ├── raw
│   ├── processed
│   ├── generate_dirty.py
│   └── clean.py
│
├── train
│   └── finetune.py
│
├── eval
│   ├── metric.py
│   └── human_eval_rubric.yaml
│
├── serve
│   ├── app.py
│   └── guardrails.py
│
├── tests
│   ├── test_clean.py
│   └── test_api.py
│
├── README.md
├── MODEL_CARD.md
├── DATA_CARD.md
├── requirements.txt
└── Dockerfile
```

---

## Pipeline Architecture

```text
Raw Data
    │
    ▼
Noise Injection
    │
    ▼
Cleaning Pipeline
    │
    ▼
Train / Validation / Test Split
    │
    ▼
LoRA Fine-Tuning
    │
    ▼
Evaluation
    │
    ▼
FastAPI Deployment
    │
    ▼
Guardrails + Monitoring
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/LLM-Fine-Tuning-and-Deployment-Pipeline.git

cd LLM-Fine-Tuning-and-Deployment-Pipeline
```

Create a virtual environment:

```bash
python -m venv llm_env
```

Activate it:

```bash
# Windows
llm_env\Scripts\activate

# Linux / macOS
source llm_env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Generate Raw Dataset

```bash
python -m data.generate_dirty
```

Creates:

```text
data/raw/dirty_data.jsonl
```

containing intentionally corrupted examples.

---

### Step 2: Clean Dataset

```bash
python -m data.clean
```

Produces:

```text
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/test.jsonl
data/processed/manifest.json
```

---

### Step 3: Fine-Tune Model

```bash
python -m train.finetune
```

Outputs:

```text
train/outputs/final/
```

along with training logs and checkpoints.

---

### Step 4: Evaluate

```bash
python -m eval.metric
```

Generates:

```text
eval/results.json
```

including perplexity, ROUGE, and token-level F1 scores.

---

### Step 5: Start API Server

```bash
uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

---

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_version": "final",
  "dataset_hash": "ea7e4e9abfc639c2"
}
```

---

### Text Generation

```http
POST /generate
```

Request:

```json
{
  "prompt": "Explain transformer attention.",
  "max_tokens": 128,
  "temperature": 0.7,
  "safe_mode": true
}
```

Response:

```json
{
  "generated_text": "...",
  "token_count": 52,
  "latency_ms": 841.23,
  "safe_mode": true
}
```

---

## Evaluation Metrics

| Metric     | Purpose                              |
| ---------- | ------------------------------------ |
| Perplexity | Measures predictive uncertainty      |
| ROUGE-1    | Unigram overlap                      |
| ROUGE-2    | Bigram overlap                       |
| ROUGE-L    | Longest common subsequence overlap   |
| Token F1   | Token-level precision/recall balance |

---

## Safety Features

The serving layer includes lightweight guardrails to demonstrate common safety practices:

* Prompt injection detection
* Jailbreak prevention
* Harmful request filtering
* Hallucination-trap detection
* PII leakage checks
* Standard refusal responses

These mechanisms are intended for demonstration and educational purposes rather than production-grade alignment.

---

## Testing

Run all tests:

```bash
pytest
```

Test coverage includes:

* Data cleaning rules
* Unicode normalization
* PII detection
* HTML filtering
* Dataset splitting
* API contract validation
* Guardrail behavior
* Rate limiting

---

## Model Card

See:

```text
MODEL_CARD.md
```

for model details, limitations, intended use cases, and deployment information.

---

## Data Card

See:

```text
DATA_CARD.md
```

for dataset generation methodology, cleaning procedures, and known limitations.

---

## Limitations

This project is intended as an educational and engineering demonstration.

* Uses a lightweight pretrained language model
* Trains on a small synthetic dataset
* Focuses on pipeline design rather than model performance
* Safety mechanisms are rule-based and not comprehensive
* Evaluation results are illustrative rather than benchmark-oriented

The primary objective is to demonstrate the end-to-end workflow of modern LLM systems rather than achieve state-of-the-art generation quality.

---

## Tech Stack

* Python
* Hugging Face Transformers
* PEFT (LoRA)
* PyTorch
* FastAPI
* Pydantic
* Structlog
* Pytest

---

## License

This project is released for educational and research purposes.

---

## Author

**Aarush Tiwari**

Computer Science & Engineering
Bennett University

Interested in Machine Learning, NLP, LLM Systems, and AI Engineering.
