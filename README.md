# 🧬 NeuroGenAI

**NeuroGenAI** is a BioAI research prototype developed by Neveroff Labs.  
It explores how **brain-inspired AI** (Spiking Neural Networks) and **genome-aware NLP models** (like DNABERT) can analyze and interpret real human DNA - including my own.

## 🚀 Project Vision

> _What if an artificial brain could read your genetic code - and visualize it?_

NeuroGenAI merges:
- 🧠 Spiking Neural Networks (SNNs)
- 🧬 Genomics & k-mer encodings
- 🤖 DNA-transformers (DNABERT)
- 📊 Protein structure visualizations (AlphaFold)
- 🧠 Hybrid NLP → spike translation


## 📂 Project Structure
```plaintext
NEUROGENAI/
├── data/              # Raw, processed, and output datasets
    ├──outputs
    ├──processed
    ├──raw
├── eda/               # Visual & interactive EDA reports
├── notebooks/         # Jupyter notebooks
├── src/               # Modular Python codebase
│   ├── data/          # Loaders for FASTA, SNPs, VCFs
│   ├── fe/            # Feature engineering (k-mers, encoders)
│   ├── ml/            # ML models (XGBoost, etc.)
│   ├── nlp/           # DNABERT embedding bridge
│   ├── snn/           # Brian2-based spike models
│   ├── viz/           # Visual tools: spike plots, GC maps
├── streamlit_app/     # Frontend demo app (coming soon)
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore
