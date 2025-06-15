# 🧬 NeuroGenAI

**NeuroGenAI** is a BioAI research prototype developed by Neveroff Labs.  
It explores how **brain-inspired AI** (Spiking Neural Networks) and **genome-aware NLP models** (like DNABERT) can analyze and interpret real human DNA.

## 🚀 Project Vision

> _What if an artificial brain could read your genetic code - and visualize it?_
![Demo](data/outputs/5. Final/visual_spikes.gif)

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
│   ├──outputs
│   ├──processed
│   ├──raw
├── eda/               # Visual & interactive EDA reports
├── notebooks/         # Jupyter notebooks
├── src/               # Modular Python codebase
│   ├── data/          # Loaders for FASTA, SNPs
│   ├── eval/          # Spikes Evaluations
│   ├── ml/            # ML models (XGBoost, etc.)
│   ├── nlp/           # DNABERT embedding bridge
│   ├── snn/           # Brian2-based spike modules
│   ├── snn_custom/    # Brian2-based spike module(custom)
│   ├── viz/           # Visual tools: spike plots, GC maps, etc.
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore