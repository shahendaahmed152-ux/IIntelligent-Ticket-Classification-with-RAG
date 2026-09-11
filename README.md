# Intelligent Support Ticket Classification & RAG Generation System

An end-to-end Machine Learning and Retrieval-Augmented Generation (RAG) system designed to classify incoming customer support tickets (Predicting `Type`, `Priority`, and `Queue`) and generate automated resolution responses using Hybrid Search and LLM completion.

---

## 🏗️ Project Structure

```text
project/
├── data/              # Raw and processed datasets
├── models/            # Saved classifiers, vectorizers, and search indexes
├── notebooks/         # Notebooks for EDA and experimentation
├── src/               # Production source modules
│   ├── classifier/    # Classification inference logic
│   ├── data/          # Text cleaning and preprocessing
│   ├── rag/           # Prompt construction and LLM pipeline
│   └── retrieval/     # BM25, FAISS, RRF, and CrossEncoder reranker
├── requirements.txt   # Dependencies
└── README.md