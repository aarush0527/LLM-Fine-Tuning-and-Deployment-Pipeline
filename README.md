# 🚀 LLM Fine-Tuning & Deployment Pipeline

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-Deployment-green)
![Pytest](https://img.shields.io/badge/Testing-Pytest-orange)

</p>

> **An end-to-end LLM engineering pipeline demonstrating data preprocessing, LoRA fine-tuning, evaluation, deployment, safety guardrails, and automated testing.**

Unlike typical fine-tuning notebooks, this project focuses on the **complete lifecycle of an LLM system** — from raw data generation to serving a model behind a production-style API.

---

## 🎯 Why This Project?

Most beginner LLM projects stop after training a model.

This repository demonstrates the engineering components required to build a usable LLM workflow:

- Data generation and preprocessing
- Dataset quality control
- Parameter-efficient fine-tuning
- Quantitative evaluation
- API deployment
- Safety guardrails
- Automated testing
- Reproducibility and monitoring

The objective is not to create a state-of-the-art model, but to showcase how modern LLM systems are built and maintained.

---

## ✨ Features

### 📦 Data Pipeline

- Synthetic dataset generation
- Noise injection for cleaning demonstrations
- Duplicate records
- HTML contamination
- Broken Unicode text
- PII-containing samples
- Boilerplate/legal text

### 🧹 Data Cleaning

- Unicode normalization
- Deduplication using hashing
- HTML filtering
- Regex-based PII detection
- Boilerplate removal
- Length-based filtering
- Deterministic dataset splitting

### 🧠 Fine-Tuning

- DistilGPT2 base model
- LoRA (Low-Rank Adaptation)
- Hugging Face Transformers
- Reproducible training configuration
- Automatic checkpointing
- Training logs and metadata

### 📊 Evaluation

- Perplexity
- ROUGE-1
- ROUGE-2
- ROUGE-L
- Token-level F1

### ⚡ Serving

- FastAPI inference API
- Structured logging
- Health monitoring
- Rate limiting
- Request validation
- Configurable generation parameters

### 🛡️ Safety

- Prompt injection detection
- Jailbreak filtering
- Hallucination-trap detection
- PII leakage protection
- Standardized refusal responses

### 🧪 Testing

- Unit tests
- Integration tests
- API contract tests
- Guardrail validation
- Reproducibility verification

---

## 🏗️ System Architecture

```text
Raw Dataset
      │
      ▼
Noise Injection
      │
      ▼
Data Cleaning Pipeline
      │
      ▼
Train / Validation / Test Split
      │
      ▼
LoRA Fine-Tuning
      │
      ▼
Model Evaluation
      │
      ▼
FastAPI Deployment
      │
      ▼
Guardrails + Monitoring
      │
      ▼
Inference API
```

---

## 📂 Project Structure

```text
.
├── data
│   ├── raw/
│   ├── processed/
│   ├── generate_dirty.py
│   └── clean.py
│
├── train/
│   └── finetune.py
│
├── eval/
│   ├── metric.py
│   └── human_eval_rubric.yaml
│
├── serve/
│   ├── app.py
│   └── guardrails.py
│
├── tests/
│   ├── test_clean.py
│   └── test_api.py
│
├── MODEL_CARD.md
├── DATA_CARD.md
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/LLM-Fine-Tuning-and-Deployment-Pipeline.git

cd LLM-Fine-Tuning-and-Deployment-Pipeline
```

Create a virtual environment:

```bash
python -m venv llm_env
```

Activate:

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

## 🚦 Quick Start

### 1. Generate Raw Data

```bash
python -m data.generate_dirty
```

Creates:

```text
data/raw/dirty_data.jsonl
```

---

### 2. Clean Dataset

```bash
python -m data.clean
```

Outputs:

```text
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/test.jsonl
data/processed/manifest.json
```

---

### 3. Fine-Tune Model

```bash
python -m train.finetune
```

Outputs:

```text
train/outputs/final/
```

---

### 4. Evaluate

```bash
python -m eval.metric
```

Outputs:

```text
eval/results.json
```

---

### 5. Launch API

```bash
uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## 🔌 API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy",
  "model_version": "final",
  "dataset_hash": "ea7e4e9abfc639c2"
}
```

---

### Generate Text

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
  "latency_ms": 842.4,
  "safe_mode": true
}
```

---

## 📈 Evaluation Metrics

| Metric | Purpose |
|----------|----------|
| Perplexity | Language modeling quality |
| ROUGE-1 | Unigram overlap |
| ROUGE-2 | Bigram overlap |
| ROUGE-L | Sequence similarity |
| Token F1 | Token-level precision & recall |

---

## 🧪 Testing

Run all tests:

```bash
pytest
```

Coverage includes:

- Cleaning pipeline validation
- Unicode normalization
- HTML filtering
- PII detection
- Dataset splitting
- API contract validation
- Rate limiting
- Guardrail enforcement

---

## 📋 Model Card

See:

```text
MODEL_CARD.md
```

for model configuration, limitations, and intended use.

---

## 📄 Data Card

See:

```text
DATA_CARD.md
```

for dataset generation methodology, preprocessing decisions, and known limitations.

---

## ⚠️ Limitations

This project is intended as an engineering demonstration.

- Uses a lightweight pretrained language model
- Trains on a small synthetic dataset
- Safety mechanisms are rule-based
- Evaluation results are illustrative
- Focuses on workflow design rather than benchmark performance

---

## 🛠️ Tech Stack

| Category | Tools |
|-----------|--------|
| Language | Python |
| Deep Learning | PyTorch |
| LLM Framework | Hugging Face Transformers |
| Fine-Tuning | PEFT (LoRA) |
| API Serving | FastAPI |
| Validation | Pydantic |
| Logging | Structlog |
| Testing | Pytest |

---

## 👨‍💻 Author

**Aarush Tiwari**

B.Tech Computer Science & Engineering  
Bennett University
