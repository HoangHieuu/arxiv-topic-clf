# ArXiv Topic Classifier

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

Classify arXiv **abstracts** into topics using classic ML models on top of **sentence embeddings** (E5/SBERT).  
The pipeline runs locally in VS Code and uses **streaming** to avoid downloading the entire dataset.

---

## Features

- **Streaming data** from `UniverseTBD/arxiv-abstracts-large` (Hugging Face).
- **Text preprocessing**: clean punctuation/digits/whitespace, lowercase, single-label filtering.
- **Sentence embeddings** via `intfloat/multilingual-e5-base` (Sentence-Transformers).
- **Models**: K-Means (majority vote), KNN, Decision Tree, Gaussian Naive Bayes.
- **Evaluation**: accuracy table, per-class precision/recall/F1, **confusion matrices** (PNG).

---

## Project Structure
arxiv-topic-clf/
├── preprocess.py # stream & filter samples, clean text, split train/test
├── vectorize.py # encode texts with E5; saves *.npy embeddings & labels
├── train_models.py # train/evaluate KMeans, KNN, DecisionTree, NaiveBayes
├── requirements-preprocess.txt
├── requirements-embed.txt
├── requirements-train.txt
├── cache/ # runtime artifacts (ignored by git)
├── .gitignore
├── LICENSE
└── README.md

> `cache/` contains generated files: JSONL splits, NumPy arrays, plots, and reports.  
> It is **ignored** by `.gitignore` and should not be committed.

---

## Quickstart

### 0) Create & activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 1) Install dependencies
```bash
pip install -r requirements-preprocess.txt
pip install -r requirements-embed.txt
pip install -r requirements-train.txt
```

### 2) Preprocess data
```bash
python preprocess.py
```

* Outputs

cache/train.jsonl, cache/test.jsonl

cache/label_to_id.json, cache/id_to_label.json

### 3) Encode with sentence embeddings (E5)
```bash
python vectorize.py
```

* Outputs

cache/X_train_emb.npy, cache/X_test_emb.npy

cache/y_train.npy, cache/y_test.npy

### 4) Train & evaluate models
```bash
python train_models.py
```

Outputs

* Console: accuracy table for all models

* cache/plots/: cm_kmeans.png, cm_knn.png, cm_decision_tree.png, cm_naive_bayes.png

* cache/reports.json: detailed per-class metrics


