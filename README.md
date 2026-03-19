# Federated Disease Detection with Advanced AI Models

## Overview

This project implements a **Federated Learning framework for skin cancer detection** using deep learning models. The system allows multiple hospitals (clients) to collaboratively train a shared, robust diagnostic model while keeping their patient data strictly private (no data sharing).

It evaluates advanced federated optimization algorithms (FedAvg, FedProx, MOON) and models (CNN, Hybrid CNN-Transformer) across multiple clinical datasets (HAM10000, ISIC).

## Key Features

- **Federated Learning Optimization**: Implements standard Federated Averaging (**FedAvg**), Proximal Optimization for Non-IID data (**FedProx**), and Model-Contrastive Federated Learning (**MOON**).
- **Advanced Architecture**: A custom **Hybrid CNN + Transformer** (ResNet50 backbone + Multi-Head Attention) to capture both local lesion features and global contextual patterns.
- **Multiple Datasets**: Supports training and evaluation on both the **HAM10000** and **ISIC** skin cancer datasets.
- **Privacy-Preserving**: Patient data remains completely local; only model weights are securely aggregated.
- **Interactive Multi-Page Streamlit App**: A professional web interface featuring image predictions, project overview, model results comparison, and federated learning visualizations.
- **Cross-Dataset Evaluation**: Test models trained on one dataset directly against other domains for true generalization metrics.

## Models Evaluated

1. **Centralized Baseline CNN** (ResNet50)
2. **Centralized Hybrid CNN-Transformer**
3. **Federated Hybrid (FedAvg)**
4. **Federated Hybrid (FedProx)**
5. **Federated Hybrid (MOON)**

## Datasets

- [**HAM10000**](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) — A large collection of multi-source dermatoscopic images of common pigmented skin lesions (highly imbalanced).
- **ISIC** — The International Skin Imaging Collaboration dataset (relatively balanced).

## Tech Stack

- **Deep Learning**: PyTorch, Torchvision
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Federated Learning**: Custom simulated environment (Non-IID splits)
- **Web App UI**: Streamlit

## Project Structure

```text
dataset/                  # Data loading, transforms, and splits (Non-IID logic)
models/                   # CNN and Hybrid model architectures
training/                 # Base training pipelines
evaluation/               # Evaluation metrics
federated/                # FedAvg, FedProx, and MOON implementations
paper/                    # LaTeX source code for the research paper
results/                  # Output directory for evaluation metrics and comparison charts
weights/                  # Saved global and local model weights

app.py                    # Multi-page Streamlit web application
train_cnn.py              # Train centralized CNN (HAM10000)
train_hybrid.py           # Train centralized Hybrid model (HAM10000)
federated_train.py        # Run all FL algorithms (HAM10000)
evaluate.py               # Evaluate and generate metrics (HAM10000)

train_cnn_isic.py         # Train centralized CNN (ISIC)
train_hybrid_isic.py      # Train centralized Hybrid (ISIC)
federated_train_isic.py   # Run all FL algorithms (ISIC)
evaluate_isic.py          # Evaluate and generate metrics (ISIC)
cross_dataset_train.py    # Cross-dataset generalization tests

generate_results.py       # Generate comparison graphs and output summaries
run_isic.ps1              # Run pipeline script for ISIC
run_rest.ps1              # Run pipeline script for HAM10000

requirements.txt          # Python dependencies
README.md                 # Project documentation
```

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Centralized Models

**For HAM10000:**
```bash
python train_cnn.py
python train_hybrid.py
```

**For ISIC:**
```bash
python train_cnn_isic.py
python train_hybrid_isic.py
```

### 3. Train Federated Learning Models (FedAvg, FedProx, MOON)

You can run individual algorithms or all at once:

**For HAM10000:**
```bash
python federated_train.py --algorithm all
```

**For ISIC:**
```bash
python federated_train_isic.py --algorithm all
```

### 4. Evaluate Models & Generate Results

```bash
python evaluate.py
python evaluate_isic.py
python generate_results.py
```

### 5. Launch the Streamlit App

Interact with the saved models, upload images, and analyze results:

```bash
streamlit run app.py
```

## License

This project is for academic and research purposes.
