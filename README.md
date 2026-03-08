# Federated Disease Detection with Advanced AI Models

## Overview

This project implements a **Federated Learning framework for skin cancer detection** using deep learning models. The system trains models across multiple simulated hospitals while keeping medical data private.

## Key Features

- Federated Learning using FedAvg
- Hybrid CNN + Transformer architecture
- Skin cancer classification using HAM10000 dataset
- Comparison between centralized and federated training
- Streamlit interface for prediction

## Models Used

1. **Centralized CNN** (ResNet50)
2. **Federated CNN**
3. **Federated Hybrid CNN + Transformer**

## Dataset

[HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) — A large collection of multi-source dermatoscopic images of common pigmented skin lesions.

## Tech Stack

- Python
- PyTorch
- Federated Learning
- Streamlit

## Project Structure

```
dataset/          # Data loading and preprocessing
models/           # CNN and Hybrid model architectures
training/         # Training utilities
evaluation/       # Evaluation metrics
federated/        # Federated learning utilities

app.py            # Streamlit web application
train_cnn.py      # Train centralized CNN
train_hybrid.py   # Train hybrid CNN + Transformer
federated_train.py # Train federated models
evaluate.py       # Evaluate trained models
generate_results.py # Generate comparison results

requirements.txt  # Python dependencies
README.md         # Project documentation
```

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the centralized CNN model

```bash
python train_cnn.py
```

### Train the hybrid model

```bash
python train_hybrid.py
```

### Train federated models

```bash
python federated_train.py
```

### Evaluate models

```bash
python evaluate.py
```

### Run the Streamlit app

```bash
streamlit run app.py
```

## License

This project is for academic and research purposes.
