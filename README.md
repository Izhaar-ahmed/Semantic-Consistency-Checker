# 🔍 Semantic Consistency Checker

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-red.svg)
![Framework](https://img.shields.io/badge/PyTorch-Transformers-orange.svg)

> A highly optimized, two-stage NLP pipeline for detecting semantic contradictions within long-form text and documents using dense passage retrieval and cross-encoder attention verification.

---

## 📖 The "why" behind the project

Imagine analyzing a 50-page legal contract. On page 2, a clause states *"The termination fee is $500."* On page 40, another clause states *"The termination fee is entirely waived."* This is a massive **semantic contradiction**. 

Finding these automatically is hard because calculating exact logical relationships (called **Natural Language Inference** or **NLI**) between two sentences requires heavy, slow neural networks. 
If we want to compare every sentence in a document against every other sentence to find contradictions, it takes $O(N^2)$ math. A 100-sentence document requires 4,950 neural network passes. A 500-sentence document requires 124,750 passes. 
It takes infinitely too long.

**This project solves that exact bottleneck.**

## 📐 Architecture: The Two-Stage Solution

We solve $O(N^2)$ by splitting the processing into an ultra-fast filter stage, and an ultra-precise verification stage.

### Stage 1: The Filter (SBERT Bi-Encoder)
* **Goal:** Massively reduce the search space by dropping sentences that aren't talking about the same thing.
* **Technology:** `sentence-transformers/all-MiniLM-L6-v2`
* **How it works:** SBERT encodes sentences individually into numerical arrays (embeddings) in microseconds. We calculate the geometric distance (Cosine Similarity) between them. If two sentences are completely unrelated (e.g. *"I ate a sandwich"* vs *"The house burned down"*), they fall below our `0.40` threshold and are instantly discarded. 
* **Result:** Turns 4,950 pairs into just ~40 highly-related candidate pairs.

### Stage 2: The Verifier (DeBERTa Cross-Encoder)
* **Goal:** Exactly determine if the related sentences agree or contradict. 
* **Technology:** `cross-encoder/nli-deberta-v3-base`
* **How it works:** The related pairs are joined together and fed into DeBERTa. Due to its dense word-to-word self-attention mechanisms, it can understand complex logic sequences. It outputs an exact margin predicting whether the pair represents `Entailment`, `Neutral`, or `Contradiction`.

---

## 🧠 Datasets & Augmentation

The neural network doesn't know what a contradiction is out of the box. We train it using specialized datasets:

1. **SNLI (Stanford Natural Language Inference) & MNLI:** Over ~80,000 human-written sentence pairs labeled with explicit entailment/contradiction tags.
2. **STS-B (Semantic Textual Similarity Benchmark):** The definitive evaluation dataset to verify that our predictions mimic true human consensus (scored 0.0 to 5.0).
3. **Synthetic Paraphrasing (T5):** We implemented `Vamsi/T5_Paraphrase_Paws` to procedurally generate "difficult" sentence rewrites out of the original datasets computationally. This forces our CrossEncoder to learn the underlying semantics of a contradiction rather than cheating by memorizing matching vocabulary words.

> **Hardware Note (Catastrophic Forgetting)**: Because we trained locally on Apple Silicon (MacOS MPS limiters), running the full million-parameter dataset would take days. We rigidly restricted sampling to 50k (SNLI) and 30k (MNLI) batches so training completed under 1 hour. This mathematically required the base DeBERTa model to "forget" massive amounts of its global baseline knowledge, resulting in lowered correlations. To improve `F1` in the future, remove limits and train on an Nvidia A100.

---

## 💻 Codebase Execution Map

This framework is highly modular. Here is exactly how to navigate the code execution logic:

* `config.py`: The control panel. Handles all model tracking, device logic (CPU vs MPS), and sets our master constraints (`SIMILARITY_THRESHOLD = 0.40`).
* `data.py`: Downloads external HuggingFace datasets and applies exact size limiters.
* `paraphraser.py`: Manages the T5 Generator architecture to create dynamic synthetic pairs.
* `biencoder.py` & `crossencoder.py`: The Stage 1 and Stage 2 wrappers mathematically structuring arrays. (Contains a highly-custom `old_fit()` manual PyTorch loop explicitly written to bypass MPS tensor crashes!).
* `pipeline.py`: The manager class that automatically connects SBERT to DeBERTa securely so models aren't instantiated multiple times in memory.
* `train.py`: Initiates the model learning cycles and writes `nli-crossencoder` models locally.
* `evaluate.py`: The benchmarking pipeline that constructs the exact correlation matrix showing our $O(N^2)$ elimination speeds natively.

---

## 🚀 Installation & Setup

You can deploy and run the entire verification suite locally on your machine.

**1. Clone the repository**
```bash
git clone https://github.com/Izhaar-ahmed/Semantic-Consistency-Checker.git
cd Semantic-Consistency-Checker
```

**2. Install dependencies**
```bash
python3 -m pip install -r requirements.txt
python3 -m pip install nltk
```

**3. Run the interactive Demo!**
```bash
python3 demo_live.py
```
*(This opens a live shell. You can type manual sentences to view exact underlying probability tensors, or type 'y' to evaluate full document paragraphs).*

---

## 📊 Quick Performance Evaluation
*(Using `SIMILARITY_THRESHOLD = 0.40` over STS-B Test Pairs)*

| Architecture | F1 Contradiction | Processing Time (sec) | Filter Reduction |
| :--- | :---: | :---: | :---: |
| Baseline 1 (SBERT Only) | 0.4195 | 10.75 | N/A |
| Baseline 2 (CE Only) | 0.4863 | 16.01 | N/A |
| **Our Two-Stage Pipeline** | **0.5827** | **1.1s (Document)** | **90.11% isolated** |

Using SBERT to drop unrelated text successfully filtered **90.11%** of the mathematical overhead away, allowing the pipeline to isolate contradictions nearly 300% faster under heavy multi-sentence document loads!

---
*Developed under MIT Open Source Architecture. Do not deploy model checkpoints to high-risk environments without rigorous cross validation.*
