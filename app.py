"""
Multi-page Streamlit Application for Federated Skin Lesion Classification.

Pages:
  1. Project Overview
  2. Image Prediction
  3. Model Results
  4. Federated Learning Visualization

Usage:
    streamlit run app.py
"""

import os
import sys
import json
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Constants & Config
# ============================================================
CLASS_NAMES = ["Benign", "Malignant"]
IMG_SIZE = 224
MODEL_PATH = "weights/hybrid_model.pth"

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Model Loading
# ============================================================
@st.cache_resource
def load_model():
    """Load the Hybrid CNN + Transformer model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model, device


def predict(image, model, device):
    """Run prediction on a single PIL image."""
    img_tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    prob = torch.sigmoid(output[0, 1]).item()

    if prob > 0.5:
        prediction = "Malignant"
    else:
        prediction = "Benign"

    return prediction, prob


def load_metrics(path):
    """Load metrics JSON file."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# ============================================================
# Page 1 — Project Overview
# ============================================================
def page_overview():
    st.markdown(
        "<h1 style='text-align:center;'>🔬 Federated Disease Detection</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='text-align:center; color:gray;'>"
        "Using Advanced AI Models for Privacy-Preserving Skin Cancer Classification"
        "</h3>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Project description
    st.subheader("📋 About This Project")
    st.markdown("""
    This project implements a **Federated Learning** framework for skin cancer detection 
    that enables multiple hospitals to collaboratively train a shared diagnostic model 
    **without exchanging patient data**.

    The system uses the **HAM10000** dermoscopy dataset and classifies skin lesions 
    into two categories: **Benign** and **Malignant**.
    """)

    # Architecture
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏗️ Model Architecture")
        st.markdown("""
        **Hybrid CNN + Transformer**

        ```
        Input Image (224×224)
              ↓
        ResNet50 Backbone
        (Feature Extraction)
              ↓
        1×1 Conv Projection
        (2048 → 512 channels)
              ↓
        Positional Encoding
              ↓
        Transformer Encoder
        (2 layers, 8 heads)
              ↓
        Global Average Pooling
              ↓
        Classification Head
              ↓
        Output: Benign / Malignant
        ```
        """)

    with col2:
        st.subheader("🏥 Federated Learning")
        st.markdown("""
        **FedAvg across 3 Hospitals**

        ```
        Global Model (Server)
              ↓
        ┌─────┼─────┐
        ↓     ↓     ↓
        H_A   H_B   H_C
        (train locally)
        ↓     ↓     ↓
        └─────┼─────┘
              ↓
        Federated Averaging
              ↓
        Updated Global Model
        ```

        **Hospital Data Distribution (Non-IID):**
        - 🏥 Hospital A: Mostly benign
        - 🏥 Hospital B: Mostly malignant
        - 🏥 Hospital C: Mixed
        """)

    st.divider()

    # Dataset info
    st.subheader("📊 Dataset — HAM10000")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", "10,015")
    col2.metric("Benign", "8,061 (80.5%)")
    col3.metric("Malignant", "1,954 (19.5%)")

    st.markdown("""
    | Category | Original Diagnoses |
    |---|---|
    | **Malignant** | Melanoma (mel), Basal Cell Carcinoma (bcc), Actinic Keratosis (akiec) |
    | **Benign** | Melanocytic Nevi (nv), Benign Keratosis (bkl), Dermatofibroma (df), Vascular (vasc) |
    """)

    # Models compared
    st.divider()
    st.subheader("🔄 Models Compared")
    st.markdown("""
    | # | Model | Type | Parameters |
    |---|---|---|---|
    | 1 | ResNet50 (Baseline) | Centralized CNN | ~23.5M |
    | 2 | Hybrid CNN + Transformer | Centralized Hybrid | ~28.8M |
    | 3 | Federated Hybrid (FedAvg) | Federated Learning | ~28.8M |
    """)


# ============================================================
# Page 2 — Image Prediction
# ============================================================
def page_prediction():
    st.markdown(
        "<h1 style='text-align:center;'>🔬 Skin Lesion Prediction</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:gray;'>"
        "Upload a dermoscopic image to classify it as Benign or Malignant "
        "using the Hybrid CNN + Transformer model."
        "</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Check model
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found: `{MODEL_PATH}`\n\n"
            "Please run `python train_hybrid.py` first."
        )
        return

    # Upload
    st.subheader("📷 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a skin lesion image...",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col_img, col_pred = st.columns([1, 1])

        with col_img:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col_pred:
            with st.spinner("Analyzing image..."):
                model, device = load_model()
                prediction, prob = predict(image, model, device)

            # Prediction result
            st.subheader("Prediction")

            if prediction == "Benign":
                st.success(f"✅ **{prediction}**")
                st.metric("Confidence", f"{(1 - prob)*100:.1f}%")
            else:
                st.error(f"⚠️ **{prediction}**")
                st.metric("Confidence", f"{prob*100:.1f}%")

            # Probability
            st.subheader("Probability")
            st.progress(prob, text=f"Malignant: {prob*100:.1f}%")

            st.caption(f"**Malignant probability:** {prob:.4f}")

        st.divider()
        st.caption(
            "⚠️ **Disclaimer**: This tool is for research purposes only. "
            "It is NOT a substitute for professional medical diagnosis."
        )


# ============================================================
# Page 3 — Model Results
# ============================================================
def page_results():
    st.markdown(
        "<h1 style='text-align:center;'>📊 Model Results</h1>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Load metrics
    cnn_metrics = load_metrics("results/centralized/metrics.json")
    hybrid_metrics = load_metrics("results/federated_cnn/metrics.json")
    fed_metrics = load_metrics("results/federated_hybrid/metrics.json")

    if not any([cnn_metrics, hybrid_metrics, fed_metrics]):
        st.warning(
            "No results found. Please run `python generate_results.py` first."
        )
        return

    # Comparison table
    st.subheader("📋 Performance Comparison")

    models_data = {}
    if cnn_metrics:
        models_data["Centralized CNN"] = cnn_metrics
    if hybrid_metrics:
        models_data["Hybrid CNN+Transformer"] = hybrid_metrics
    if fed_metrics:
        models_data["Federated Hybrid (FedAvg)"] = fed_metrics

    # Build table
    table_md = "| Metric | " + " | ".join(models_data.keys()) + " |\n"
    table_md += "|---|" + "|".join(["---"] * len(models_data)) + "|\n"

    for metric_key, metric_label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1 Score"),
        ("auc_roc", "AUC-ROC"),
    ]:
        row = f"| **{metric_label}** |"
        for data in models_data.values():
            val = data.get(metric_key, 0) * 100
            row += f" {val:.2f}% |"
        table_md += row + "\n"

    st.markdown(table_md)

    # Key metrics cards
    if fed_metrics:
        st.subheader("🏆 Best Model — Federated Hybrid (FedAvg)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{fed_metrics['accuracy']*100:.1f}%")
        c2.metric("Precision", f"{fed_metrics['precision']*100:.1f}%")
        c3.metric("Recall", f"{fed_metrics['recall']*100:.1f}%")
        c4.metric("F1 Score", f"{fed_metrics['f1_score']*100:.1f}%")
        c5.metric("AUC-ROC", f"{fed_metrics['auc_roc']*100:.1f}%")

    st.divider()

    # Plots
    st.subheader("📈 Comparison Charts")

    col1, col2 = st.columns(2)

    # Metric comparison bar chart
    if os.path.exists("results/comparison/metric_comparison.png"):
        with col1:
            st.image("results/comparison/metric_comparison.png",
                     caption="Model Performance Comparison",
                     use_container_width=True)

    # ROC comparison
    if os.path.exists("results/comparison/roc_comparison.png"):
        with col2:
            st.image("results/comparison/roc_comparison.png",
                     caption="ROC Curve Comparison",
                     use_container_width=True)

    # Training curves
    if os.path.exists("results/comparison/training_curves.png"):
        st.image("results/comparison/training_curves.png",
                 caption="Training Accuracy & Loss Curves",
                 use_container_width=True)

    st.divider()

    # Confusion matrices
    st.subheader("🔢 Confusion Matrices")
    cm_cols = st.columns(3)

    cm_files = [
        ("results/centralized/confusion_matrix.png", "Centralized CNN"),
        ("results/federated_cnn/confusion_matrix.png", "Hybrid CNN+Transformer"),
        ("results/federated_hybrid/confusion_matrix.png", "Federated Hybrid"),
    ]

    for i, (path, label) in enumerate(cm_files):
        if os.path.exists(path):
            with cm_cols[i]:
                st.image(path, caption=label, use_container_width=True)

    st.divider()

    # Individual ROC curves
    st.subheader("📉 Individual ROC Curves")
    roc_cols = st.columns(3)

    roc_files = [
        ("results/centralized/roc_curve.png", "Centralized CNN"),
        ("results/federated_cnn/roc_curve.png", "Hybrid CNN+Transformer"),
        ("results/federated_hybrid/roc_curve.png", "Federated Hybrid"),
    ]

    for i, (path, label) in enumerate(roc_files):
        if os.path.exists(path):
            with roc_cols[i]:
                st.image(path, caption=label, use_container_width=True)


# ============================================================
# Page 4 — Federated Learning Visualization
# ============================================================
def page_federated():
    st.markdown(
        "<h1 style='text-align:center;'>🏥 Federated Learning Visualization</h1>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Federated overview
    st.subheader("📋 Federated Training Setup")

    col1, col2, col3 = st.columns(3)
    col1.metric("Hospitals", "3")
    col2.metric("Algorithm", "FedAvg")
    col3.metric("Communication Rounds", "5")

    st.markdown("""
    | Parameter | Value |
    |---|---|
    | Model | Hybrid CNN + Transformer |
    | Total Parameters | ~28.8M |
    | Batch Size | 32 |
    | Learning Rate | 0.0001 |
    | Local Epochs per Round | 2 |
    | Optimizer | Adam |
    """)

    st.divider()

    # Hospital data distribution
    st.subheader("🏥 Hospital Data Distribution (Non-IID)")

    hosp_col1, hosp_col2, hosp_col3 = st.columns(3)

    with hosp_col1:
        st.markdown("### Hospital A")
        st.markdown("**General Practice**")
        st.markdown("- 🟢 Mostly **Benign**")
        st.markdown("- ~80% benign, ~20% malignant")

    with hosp_col2:
        st.markdown("### Hospital B")
        st.markdown("**Oncology Center**")
        st.markdown("- 🔴 Mostly **Malignant**")
        st.markdown("- ~40% benign, ~60% malignant")

    with hosp_col3:
        st.markdown("### Hospital C")
        st.markdown("**Mixed Clinic**")
        st.markdown("- 🟡 **Mixed** distribution")
        st.markdown("- ~55% benign, ~45% malignant")

    st.divider()

    # Convergence plot
    st.subheader("📈 Federated Convergence")

    if os.path.exists("results/federated_hybrid/federated_convergence.png"):
        st.image(
            "results/federated_hybrid/federated_convergence.png",
            caption="Federated Training Convergence Across Rounds",
            use_container_width=True,
        )
    else:
        st.info("Convergence plot not found. Run `python generate_results.py`.")

    st.divider()

    # FedAvg explanation
    st.subheader("🔄 FedAvg Algorithm")
    st.markdown("""
    **Federated Averaging (FedAvg)** process:

    ```
    For each round t = 1, 2, ..., T:
        1. Server sends global model to all clients
        2. Each client trains locally for E epochs
        3. Clients send updated weights back to server
        4. Server averages all client weights:
           w_global = (1/K) × Σ w_k
        5. Updated global model is distributed
    ```

    **Key advantage:** Patient data **never leaves** the hospital.
    Only model weight updates are communicated.
    """)

    # Results
    st.divider()
    st.subheader("🏆 Federated Model Results")

    fed_metrics = load_metrics("results/federated_hybrid/metrics.json")
    if fed_metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{fed_metrics['accuracy']*100:.1f}%")
        c2.metric("Precision", f"{fed_metrics['precision']*100:.1f}%")
        c3.metric("Recall", f"{fed_metrics['recall']*100:.1f}%")
        c4.metric("F1 Score", f"{fed_metrics['f1_score']*100:.1f}%")
        c5.metric("AUC-ROC", f"{fed_metrics['auc_roc']*100:.1f}%")


# ============================================================
# Main — Navigation
# ============================================================
def main():
    st.set_page_config(
        page_title="Federated Skin Lesion Classifier",
        page_icon="🔬",
        layout="wide",
    )

    # Sidebar navigation
    st.sidebar.title("🔬 Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "📋 Project Overview",
            "🔍 Image Prediction",
            "📊 Model Results",
            "🏥 Federated Learning",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption("Federated Disease Detection")
    st.sidebar.caption("Using Advanced AI Models")

    # Route to page
    if page == "📋 Project Overview":
        page_overview()
    elif page == "🔍 Image Prediction":
        page_prediction()
    elif page == "📊 Model Results":
        page_results()
    elif page == "🏥 Federated Learning":
        page_federated()


if __name__ == "__main__":
    main()
